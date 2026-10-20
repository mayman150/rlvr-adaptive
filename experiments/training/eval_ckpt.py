"""Evaluate saved checkpoints: sampled pass@k (n answers per problem, temperature 1) and,
optionally, greedy accuracy. Writes per-problem counts and answer texts.

Usage: python eval_ckpt.py --ckpts DIR [--base MODEL] --problems FILE --n 32 --out DIR
       [--greedy-problems FILE] [--greedy-steps 300] [--limit N] [--seed S]
DIR holds ckpt_step<N>/ folders (saved by train_grpo.py --save-steps).
"""

import argparse
import gzip
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_grpo import build_prompt, pass_at_k, stop_token_ids  # noqa: E402
from verifier import last_boxed, reward  # noqa: E402


def load_problems(path, limit):
    p = Path(path) if Path(path).exists() else HERE / "data" / path
    return json.loads(p.read_text())[:limit]


@torch.no_grad()
def generate(model, tok, stop, texts, n_each, max_new, seed, greedy, gen_batch, prefill_tokens, temperature=1.0, meta=None):
    """Decoded answers; if `meta` is a list, (token length, ended with a stop token) is appended per answer."""
    items = [t for t in texts for _ in range(n_each)]
    outs, s, budget = [], 0, gen_batch * prefill_tokens
    torch.manual_seed(seed)
    pad = tok.pad_token_id if tok.pad_token_id is not None else stop[0]
    while s < len(items):
        b = gen_batch
        enc = tok(items[s:s + b], return_tensors="pt", padding=True)
        while b > 1 and enc.input_ids.shape[1] * b > budget:
            b = max(1, budget // enc.input_ids.shape[1])
            enc = tok(items[s:s + b], return_tensors="pt", padding=True)
        enc = enc.to(model.device)
        kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        try:
            gen = model.generate(**enc, max_new_tokens=max_new, eos_token_id=stop, pad_token_id=pad, **kw)
        except torch.OutOfMemoryError:
            if b == 1:
                raise
            del enc
            torch.cuda.empty_cache()
            budget = max(1, budget // 2)
            print(f"generation out of memory at batch {b}; retrying with a smaller batch", flush=True)
            continue
        for row in gen[:, enc.input_ids.shape[1]:]:
            ids = row.tolist()
            cut = next((i + 1 for i, t in enumerate(ids) if t in stop), None)
            if meta is not None:
                meta.append((cut if cut is not None else len([t for t in ids if t != pad]), cut is not None))
            outs.append(tok.decode(ids[:cut or len(ids)], skip_special_tokens=True))
        s += b
    return outs


@torch.no_grad()
def prefix_entropy(model, tok, sequences, temperature, device, batch=8):
    """Mean per-token entropy of softmax(logits / temperature) over the response tokens of fixed
    (frozen) prompt-response sequences, so that models are compared on the same prefixes. Token-weighted,
    on retokenized text (the saved texts lack the stop token), responses cut at 512 tokens; it therefore
    differs from the training-time entropy probe, which uses each model's own samples."""
    total, count = 0.0, 0
    for s in range(0, len(sequences), batch):
        chunk = sequences[s:s + batch]
        prompts = [build_prompt(tok, c["question"]) for c in chunk]
        p_ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
        r_ids = [tok(c["response"], add_special_tokens=False).input_ids[:512] for c in chunk]
        seqs = [a + b for a, b in zip(p_ids, r_ids)]
        width = max(len(x) for x in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.tensor([x + [pad] * (width - len(x)) for x in seqs], device=device)
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in seqs], device=device)
        logits = model(input_ids=ids, attention_mask=mask).logits.float() / temperature
        logp = torch.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(-1)  # entropy of the next-token distribution at each position
        for i, (a, b) in enumerate(zip(p_ids, r_ids)):
            # positions len(a)-1 .. len(a)+len(b)-2 predict the response tokens
            e = ent[i, len(a) - 1:len(a) + len(b) - 1]
            total += float(e.sum())
            count += e.numel()
        del logits, logp, ent
    return total / max(count, 1)


def evaluate(model_path, tag, args, problems, greedy_problems, out_dir, sampled=True, temperature=1.0, prefixes=None):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(args.device).eval()
    stop = stop_token_ids(tok)
    res = {"tag": tag, "model": str(model_path), "n": args.n, "problems": len(problems), "pass_at_k": {},
           "temperature": temperature}
    if prefixes is not None:
        res["prefix_entropy"] = prefix_entropy(model, tok, prefixes, temperature, args.device)
    if sampled:
        texts = [build_prompt(tok, q["question"]) for q in problems]
        meta = []
        outs = generate(model, tok, stop, texts, args.n, args.max_new, args.seed, False, args.gen_batch, args.prefill_tokens,
                        temperature, meta)
        res["mean_length_chars"] = sum(len(o) for o in outs) / len(outs)
        res["mean_length_tokens"] = sum(m[0] for m in meta) / len(meta)
        res["truncated_rate"] = sum(not m[1] for m in meta) / len(meta)
        res["boxed_rate"] = sum(last_boxed(o) is not None for o in outs) / len(outs)  # parseable final box
        correct = [reward(o, problems[i // args.n]["answer"]) for i, o in enumerate(outs)]
        counts = [sum(correct[i * args.n:(i + 1) * args.n]) for i in range(len(problems))]
        ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= args.n]
        res.update({"counts": counts, "pass_at_k": {k: sum(pass_at_k(args.n, c, k) for c in counts) / len(counts) for k in ks}})
        with gzip.open(out_dir / f"{tag}_texts.json.gz", "wt") as fh:
            json.dump({"step": tag, "n": args.n, "problems": [{"question": q["question"], "answer": q["answer"]} for q in problems],
                       "samples": [{"problem": i // args.n, "text": o, "reward": c} for i, (o, c) in enumerate(zip(outs, correct))]}, fh)
    if greedy_problems is not None:
        gt = [build_prompt(tok, q["question"]) for q in greedy_problems]
        go = generate(model, tok, stop, gt, 1, args.max_new, args.seed, True, args.gen_batch, args.prefill_tokens)
        gc = [reward(o, q["answer"]) for o, q in zip(go, greedy_problems)]
        res["greedy"] = {"accuracy": sum(gc) / len(gc), "n": len(gc), "correct": gc}
    res["seconds"] = time.time() - t0
    (out_dir / f"{tag}_eval.json").write_text(json.dumps(res))
    print(f"{tag}: pass@1 {100 * res['pass_at_k'].get(1, math.nan):.1f} pass@16 {100 * res['pass_at_k'].get(16, math.nan):.1f} "
          f"pass@32 {100 * res['pass_at_k'].get(32, math.nan):.1f}"
          + (f" greedy {100 * res['greedy']['accuracy']:.1f}" if "greedy" in res else "") + f" ({res['seconds']:.0f}s)", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--base", default=None, help="also evaluate this model as step 0")
    ap.add_argument("--problems", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--greedy-problems", default=None)
    ap.add_argument("--greedy-steps", default="", help="checkpoint steps that also get the greedy evaluation")
    ap.add_argument("--sampled-steps", default="", help="checkpoint steps that get the sampled evaluation (default all)")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--prefill-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expect-steps", default="", help="fail unless checkpoints (or the base, as 0) exist for these steps")
    ap.add_argument("--temperatures", default="", help="sampling temperatures (tags become step<N>_T<t>); default 1.0 only")
    ap.add_argument("--entropy-prefixes", default=None, help="frozen prompt-response sequences for prefix entropy")
    ap.add_argument("--base-temperatures", default="", help="temperatures for the base model (default: --temperatures)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_problems(args.problems, args.limit)
    greedy_problems = load_problems(args.greedy_problems, None) if args.greedy_problems else None
    greedy_steps = {int(s) for s in args.greedy_steps.split(",") if s}
    sampled_steps = {int(s) for s in args.sampled_steps.split(",") if s}
    jobs = []
    if args.base:
        jobs.append((args.base, "step0", 0))
    for d in sorted(Path(args.ckpts).glob("ckpt_step*"), key=lambda d: int(re.findall(r"\d+", d.name)[-1])):
        step = int(re.findall(r"\d+", d.name)[-1])
        jobs.append((d, f"step{step}", step))
    expected = {int(s) for s in args.expect_steps.split(",") if s}
    found = {step for _, _, step in jobs}
    if expected - found:
        raise SystemExit(f"missing checkpoints for steps {sorted(expected - found)} under {args.ckpts}")
    temps = [float(t) for t in args.temperatures.split(",") if t] or [None]
    base_temps = [float(t) for t in args.base_temperatures.split(",") if t] or temps
    prefixes = load_problems(args.entropy_prefixes, None) if args.entropy_prefixes else None
    for path, tag, step in jobs:
        sampled = not sampled_steps or step in sampled_steps
        if not sampled and step not in greedy_steps:
            continue
        for t in (base_temps if (args.base and step == 0 and path == args.base) else temps):
            full = tag if t is None else f"{tag}_T{t:.2f}"
            if (out_dir / f"{full}_eval.json").exists():
                print(f"{full}: already done", flush=True)
                continue
            greedy_here = step in greedy_steps and (t is None or abs(t - 1.0) < 1e-9)  # greedy once per checkpoint
            evaluate(path, full, args, problems, greedy_problems if greedy_here else None, out_dir, sampled,
                     1.0 if t is None else t, prefixes)
    print("done", flush=True)


if __name__ == "__main__":
    main()
