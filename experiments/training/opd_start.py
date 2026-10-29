"""Starting model for Pre-registration 34: on-policy distillation (OPD) of the base from a teacher before RL.

The student samples one answer per problem at its own current policy (T = 1, up to max_new tokens, stop at the stop
tokens); the teacher scores the same prompt-plus-answer; the loss is the per-token reverse KL between the student's
and the teacher's full next-token distributions at the student's prefixes,
    KL(pi_s(. | prefix) || pi_t(. | prefix)) = sum_v pi_s(v) (log pi_s(v) - log pi_t(v)),
averaged over the batch's response tokens (gradient through the student only). Student and teacher share the
vocabulary (the teacher's extra rows beyond the student's vocabulary, if any, are dropped).
Problems are visited uniformly in shuffled passes (as in sft_start.py); optimizer and batch as sft_start.py
(AdamW 1e-5, betas 0.9/0.99, no weight decay, batches of 8, clip 1). Milestones count presented (generated) response
tokens. Usage:
python opd_start.py --model BASE --teacher DIR --problems sel.json --milestones 150000 --out DIR
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_grpo import Runner, build_prompt, save_bf16  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from verifier import reward  # noqa: E402


def logits_of(model, ids, autocast):
    with autocast:
        return model(input_ids=ids[None]).logits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--problems", required=True, help="JSON list of the selection's questions")
    ap.add_argument("--milestones", default="150000")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3434)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runner = Runner(argparse.Namespace(model=a.model, device="cuda", gen_batch=a.batch, prefill_tokens=4096,
                                       max_new=a.max_new, train_batch=1))
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16).to("cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    train = {q["question"]: q for q in json.loads((HERE / "data" / "train.json").read_text())}
    sel = Path(a.problems) if Path(a.problems).exists() else HERE / "data" / a.problems
    questions = json.loads(sel.read_text())
    if len(set(questions)) != len(questions) or any(q not in train for q in questions):
        raise SystemExit("selection has duplicates or questions outside train.json")
    tok = runner.tok
    prompts = [build_prompt(tok, q) for q in questions]
    vs = runner.model.get_output_embeddings().weight.shape[0]
    milestones = sorted(int(x) for x in a.milestones.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    opt = torch.optim.AdamW(runner.params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    order, presented, steps, log, mi, visits = [], 0, 0, [], 0, [0] * len(questions)
    while mi < len(milestones):
        while len(order) < a.batch:
            nxt = list(range(len(questions)))
            rng.shuffle(nxt)
            order += nxt
        batch = [order.pop(0) for _ in range(a.batch)]
        for i in batch:
            visits[i] += 1
        gens = runner.generate([prompts[i] for i in batch], 1, a.max_new, a.seed * 100003 + steps)
        runner.model.train()
        n_tok = sum(len(r) for _, r in gens)
        opt.zero_grad(set_to_none=True)
        kl_sum, correct = 0.0, 0
        for (p_ids, r_ids), i in zip(gens, batch):
            correct += reward(tok.decode(r_ids, skip_special_tokens=True), train[questions[i]]["answer"])
            if len(r_ids) == 0:
                continue
            ids = torch.cat([p_ids, r_ids]).to("cuda")
            lo, hi = len(p_ids) - 1, len(p_ids) - 1 + len(r_ids)
            with torch.no_grad():
                t_lp = F.log_softmax(logits_of(teacher, ids, runner.autocast)[lo:hi, :vs].float(), dim=-1)
            s_lp = F.log_softmax(logits_of(runner.model, ids, runner.autocast)[lo:hi].float(), dim=-1)
            kl = (s_lp.exp() * (s_lp - t_lp)).sum(-1)  # per-token reverse KL
            (kl.sum() / n_tok).backward()
            kl_sum += float(kl.detach().sum())
            del t_lp, s_lp, kl
        gn = float(torch.nn.utils.clip_grad_norm_(runner.params, a.clip))
        if not math.isfinite(gn):
            raise SystemExit(f"non-finite gradient norm at OPD step {steps + 1}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps += 1
        presented += n_tok
        log.append({"step": steps, "reverse_kl_per_token": kl_sum / max(n_tok, 1), "presented": presented,
                    "batch_reward": correct / len(batch), "grad_norm": gn, "mean_length": n_tok / len(batch),
                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9})
        if steps % 5 == 0:
            print(f"opd step {steps} reverse KL {kl_sum / max(n_tok, 1):.4f} reward {correct / len(batch):.3f} "
                  f"presented {presented}", flush=True)
        while mi < len(milestones) and presented >= milestones[mi]:
            path = out / f"start_{milestones[mi]}"
            save_bf16(runner, path)
            h = hashlib.sha256()
            for f in sorted(path.glob("*.safetensors")):
                h.update(f.read_bytes())
            meta = {"milestone": milestones[mi], "presented_tokens": presented, "steps": steps, "problems": len(questions),
                    "problem_visits": sum(visits), "teacher": a.teacher, "objective": "reverse KL, full distributions, student prefixes",
                    "sha256": h.hexdigest()}
            (path / "start_meta.json").write_text(json.dumps(meta, indent=1))
            print(f"saved {path}: {meta}", flush=True)
            mi += 1
    (out / "opd_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
