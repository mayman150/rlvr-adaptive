"""Starting models for Pre-registration 30: supervised fine-tuning of the base on a selection's teacher solutions,
saved at token milestones along one trajectory (fixed SFT seed, independent of any RL seed).

Data: a discovery-format archive (question, answer, responses) restricted to the selection's problems (--problems, a
JSON list of questions). Every distinct response that survives Pre-registration 14's canonicalization (retokenized
without special tokens, at most max_new - 1 tokens, verifies, final box; stop token appended) is kept. Order: problems
are visited uniformly in shuffled passes; each visit uses the problem's next solution in rotation (so problems with
more teacher successes are not overweighted).
Loss: mean token NLL of the response (prompt masked), AdamW (lr 1e-5, betas 0.9/0.99, no weight decay), batches of 8,
clip 1 (Pre-registration 14's warm start). --tail: TailSFT (arXiv 2608.25756, Algorithm 1): each example's
initial-model mean NLL l0 is precomputed; at each encounter the margin m = l_t - l0 is computed and the fraction f of
the batch with the most negative margins (most improved) is dropped, f rising linearly from 0 to 0.5 over training;
the loss is the summed NLL of the retained examples divided by their token count. Adaptation: the selection batch is
the optimizer batch of 8 (the paper uses 16 per forward pass); the dropped count is floor(f * 8), with f computed after counting the batch's tokens (it reaches 0.5 on the final
batch); ties keep the lower index.
Milestones count presented response tokens (retained tokens are also reported).
Usage: python sft_start.py --model BASE --archive A.json --problems sel.json --milestones 500000,2000000 --out DIR [--tail]
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_grpo import Runner, build_prompt, save_bf16, stop_token_ids  # noqa: E402
from verifier import last_boxed, reward  # noqa: E402


def marker_mask(tok, resp, ids):
    """1 for response tokens lying entirely inside the last final-answer box's opening "\\boxed{" or its matching
    closing brace (Pre-registration 35's format-only control); the stop token appended later is also supervised."""
    start = resp.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 1, start + len("\\boxed{")
    while i < len(resp) and depth:
        depth += {"{": 1, "}": -1}.get(resp[i], 0)
        i += 1
    if depth:
        return None
    spans = [(start, start + len("\\boxed{")), (i - 1, i)]
    offs = tok(resp, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    assert len(offs) == len(ids)
    # supervise only tokens lying entirely inside a marker span: a token that merges a brace with answer or
    # reasoning characters (e.g. "{x" or "}}") is not supervised
    m = [int(b > a and any(s0 <= a and b <= e for s0, e in spans)) for a, b in offs]
    answer = (start + len("\\boxed{"), i - 1)
    assert not any(k and a < answer[1] and b > answer[0] for k, (a, b) in zip(m, offs)), "a supervised token overlaps the answer"
    return m


def examples(runner, archive, questions, max_new, verify="correct", loss_mask="all"):
    tok = runner.tok
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    stop = im_end if isinstance(im_end, int) and im_end in runner.stop_ids else runner.stop_ids[0]
    raw = json.loads(Path(archive).read_text())
    recs = {d["question"]: d for d in raw}
    if len(recs) != len(raw):
        raise SystemExit(f"{archive}: duplicate archive records")
    if len(set(questions)) != len(questions):
        raise SystemExit("duplicate questions in the selection")
    missing = [q for q in questions if q not in recs]
    if missing:
        raise SystemExit(f"{len(missing)} selected questions are not in the archive")
    out, dropped = [], {"too_long": 0, "unverified": 0, "duplicate": 0, "no_solution": 0, "no_marker": 0}
    for q in questions:
        d = recs[q]
        sols, seen = [], set()
        for resp in d.get("responses") or ([d["response"]] if d.get("response") else []):
            ids = tok(resp, add_special_tokens=False).input_ids
            text = tok.decode(ids, skip_special_tokens=True)
            if len(ids) > max_new - 1:
                dropped["too_long"] += 1
            elif last_boxed(text) is None or (verify == "correct" and not reward(text, d["answer"])):
                dropped["unverified"] += 1
            elif tuple(ids) in seen:
                dropped["duplicate"] += 1
            else:
                m = marker_mask(tok, resp, ids) if loss_mask == "format" else None
                if loss_mask == "format" and (m is None or not any(m)):
                    dropped["no_marker"] += 1
                    continue
                seen.add(tuple(ids))
                t = torch.tensor(ids + [stop])
                if m is not None:
                    t.mask = torch.tensor(m + [1], dtype=torch.float32)  # marker tokens and the stop token
                sols.append(t)
        if sols:
            out.append((torch.tensor(tok(build_prompt(tok, q)).input_ids), sols))
        else:
            dropped["no_solution"] += 1
    if dropped["no_solution"]:  # selections are built from problems with a surviving solution: a loss breaks matching
        raise SystemExit(f"{dropped['no_solution']} selected problems have no surviving solution")
    return out, dropped


def token_nll(runner, pairs, masks=None):
    """Per-example (summed NLL, token count) of the responses; with masks, over the masked tokens only."""
    lps = runner.token_logprobs(pairs)
    if masks is None:
        return [(-lp.sum(), lp.numel()) for lp in lps]
    return [(-(lp * m.to(lp.device)).sum(), int(m.sum())) for lp, m in zip(lps, masks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--problems", required=True, help="JSON list of the selection's questions")
    ap.add_argument("--milestones", default="500000,2000000")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3030)
    ap.add_argument("--tail", action="store_true", help="TailSFT filtering")
    ap.add_argument("--tail-max-drop", type=float, default=0.5)
    ap.add_argument("--verify", default="correct", choices=("correct", "boxed"),
                    help="keep verifier-correct responses (default) or any response with a final box (P35 controls)")
    ap.add_argument("--loss-mask", default="all", choices=("all", "format"),
                    help="supervise every response token (default) or only the final box's markers and the stop token")
    ap.add_argument("--weights", default=None, help="JSON {question: weight}: draw problems with replacement in proportion "
                    "to the weight (Pre-registration 36's coverage-targeted stream) instead of shuffled passes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runner = Runner(argparse.Namespace(model=a.model, device="cuda", gen_batch=8, prefill_tokens=256, max_new=a.max_new, train_batch=a.batch))
    p = Path(a.problems) if Path(a.problems).exists() else HERE / "data" / a.problems
    arch = Path(a.archive) if Path(a.archive).exists() else HERE / "data" / a.archive
    questions = json.loads(p.read_text())
    data, dropped = examples(runner, arch, questions, a.max_new, a.verify, a.loss_mask)
    wts = None
    if a.weights:
        wp = Path(a.weights) if Path(a.weights).exists() else HERE / "data" / a.weights
        wmap = json.loads(wp.read_text())
        wts = [float(wmap[q]) for q in questions]  # data[i] belongs to questions[i] (a problem without a solution aborts)
        if not all(math.isfinite(w) and w >= 0 for w in wts) or sum(wts) <= 0:
            raise SystemExit("invalid weights")
    if a.loss_mask == "format" and a.tail:
        raise SystemExit("--loss-mask format is not combined with --tail")
    if not (a.batch >= 1 and a.lr > 0 and 0 <= a.tail_max_drop < 1 and data):
        raise SystemExit("invalid settings or empty selection")
    milestones = sorted(int(x) for x in a.milestones.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    opt = torch.optim.AdamW(runner.params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    runner.model.train()
    l0 = {}
    if a.tail:  # initial-model mean NLL of every (problem, solution) pair
        with torch.no_grad():
            pairs = [(i, j) for i, (_, sols) in enumerate(data) for j in range(len(sols))]
            for s in range(0, len(pairs), a.batch):
                chunk = pairs[s:s + a.batch]
                res = token_nll(runner, [(data[i][0].to("cuda"), data[i][1][j].to("cuda")) for i, j in chunk])
                for (i, j), (nll, n) in zip(chunk, res):
                    l0[(i, j)] = float(nll) / n
    rot = [0] * len(data)
    order, presented, retained, steps, log, mi = [], 0, 0, 0, [], 0
    total = milestones[-1]
    while mi < len(milestones):
        while wts is not None and len(order) < a.batch:  # weighted draws with replacement
            order += rng.choices(range(len(data)), weights=wts, k=256)
        while len(order) < a.batch:  # refill with shuffled passes (also for selections smaller than a batch)
            nxt = list(range(len(data)))
            rng.shuffle(nxt)
            order += nxt
        batch = [order.pop(0) for _ in range(a.batch)]
        items = []
        for i in batch:
            j = rot[i] % len(data[i][1])
            rot[i] += 1
            items.append((i, j))
        masks = [data[i][1][j].mask for i, j in items] if a.loss_mask == "format" else None
        res = token_nll(runner, [(data[i][0].to("cuda"), data[i][1][j].to("cuda")) for i, j in items], masks)
        lengths = [len(data[i][1][j]) for i, j in items]  # presented response tokens (the dose), mask or not
        keep = list(range(len(items)))
        if a.tail:
            # progress counts this batch's tokens, so the schedule reaches max_drop on the final batch
            frac = a.tail_max_drop * min(1.0, (presented + sum(lengths)) / total)
            n_drop = math.floor(frac * len(items) + 1e-9)
            margins = [float(nll) / n - l0[it] for (nll, n), it in zip(res, items)]
            drop = sorted(range(len(items)), key=lambda k: (margins[k], -k))[:n_drop]  # ties: the lower index is kept
            keep = [k for k in keep if k not in set(drop)]
            loss = sum(res[k][0] for k in keep) / max(sum(res[k][1] for k in keep), 1)
        else:
            loss = sum(nll for nll, _ in res) / sum(n for _, n in res)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(runner.params, a.clip))
        if not math.isfinite(gn):  # a skipped update must not count toward the milestones
            raise SystemExit(f"non-finite gradient norm at SFT step {steps + 1}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps += 1
        presented += sum(lengths)
        retained += sum(res[k][1] for k in keep)  # supervised tokens (equal to presented without TailSFT or masks)
        if steps % 20 == 0:
            log.append({"step": steps, "loss": float(loss), "presented": presented, "retained": retained, "grad_norm": gn,
                        "peak_gb": torch.cuda.max_memory_allocated() / 1e9})
            print(f"sft step {steps} loss {float(loss):.4f} presented {presented} retained {retained}", flush=True)
        while mi < len(milestones) and presented >= milestones[mi]:
            path = out / f"start_{milestones[mi]}"
            save_bf16(runner, path)
            runner.tok.save_pretrained(path)
            h = hashlib.sha256()
            for f in sorted(path.glob("*.safetensors")):
                h.update(f.read_bytes())
            uniq = sum(min(rot[i], len(data[i][1])) for i in range(len(data)))
            meta = {"milestone": milestones[mi], "presented_tokens": presented, "retained_tokens": retained, "steps": steps,
                    "problems": len(data), "solutions": sum(len(s) for _, s in data), "unique_solutions_seen": uniq,
                    "problem_visits": sum(rot), "dropped_at_load": dropped, "tail": a.tail, "sha256": h.hexdigest()}
            (path / "start_meta.json").write_text(json.dumps(meta, indent=1))
            print(f"saved {path}: {meta}", flush=True)
            mi += 1
    (out / "sft_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
