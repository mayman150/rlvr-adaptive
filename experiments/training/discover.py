"""Discovery for Pre-registration 14: sample the base model n times per training question and keep
the first verified, stop-terminated answer with a parseable final box; record the success count.

Usage: python discover.py --model M --problems train.json --n 32 --shard 0 --shards 4 --out DIR
Writes DIR/discovery_shard<k>.json: [{"question", "answer", "c", "n", "response" (or null)}].
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_ckpt import generate, load_problems  # noqa: E402
from train_grpo import build_prompt, stop_token_ids  # noqa: E402
from verifier import last_boxed, reward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", default="train.json")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--prefill-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1414)
    ap.add_argument("--chunk", type=int, default=64, help="questions per generation call (results saved after each)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"discovery_shard{args.shard}.json"
    problems = load_problems(args.problems, None)[args.shard::args.shards]
    done = json.loads(path.read_text()) if path.exists() else []
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    stop = stop_token_ids(tok)
    t0 = time.time()
    for s in range(len(done), len(problems), args.chunk):
        chunk = problems[s:s + args.chunk]
        texts = [build_prompt(tok, q["question"]) for q in chunk]
        meta = []
        outs = generate(model, tok, stop, texts, args.n, args.max_new, args.seed + s, False, args.gen_batch,
                        args.prefill_tokens, 1.0, meta)
        for i, q in enumerate(chunk):
            answers = outs[i * args.n:(i + 1) * args.n]
            ends = meta[i * args.n:(i + 1) * args.n]
            rewards = [reward(a, q["answer"]) for a in answers]
            pick = next((a for a, r, (_, stopped) in zip(answers, rewards, ends)
                         if r and stopped and last_boxed(a) is not None), None)
            done.append({"question": q["question"], "answer": q["answer"], "c": int(sum(rewards)), "n": args.n,
                         "response": pick})
        path.write_text(json.dumps(done))
        print(f"shard {args.shard}: {len(done)}/{len(problems)} questions, {sum(d['response'] is not None for d in done)} with a "
              f"retained success ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
