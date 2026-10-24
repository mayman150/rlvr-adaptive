"""Pre-registration 19: retained supervised solutions per seed after the trainer's checks, by base-count stratum.

Replicates train_grpo.py exactly (seeded shuffle, last --calib-prompts problems withheld, load_discovery's
retokenization, length limit max_new - 1 and re-verification) with the tokenizer only.
Usage: python p19_retained.py --model DIR [--seeds 8,9,10] [--max-new 512]
"""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from train_grpo import HERE, build_prompt  # noqa: F401  (same module state as the trainer)
from verifier import last_boxed, reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="8,9,10")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--calib-prompts", type=int, default=16)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    base = json.loads((HERE / "data" / "train.json").read_text())
    out = {}
    for name in ("human_positive", "human_swap", "human_solvable"):
        found = {d["question"]: d for d in json.loads((HERE / "data" / f"{name}.json").read_text())}
        for seed in [int(s) for s in args.seeds.split(",")]:
            problems = list(base)
            random.Random(seed).shuffle(problems)
            problems = problems[: -args.calib_prompts]
            stats = {"c0": [0, 0], "c>0": [0, 0], "tokens": 0, "too_long": 0, "unverified": 0}
            for q in problems:
                d = found[q["question"]]
                if not d.get("response"):
                    continue
                key = "c0" if d["c"] == 0 else "c>0"
                stats[key][0] += 1
                ids = tok(d["response"], add_special_tokens=False).input_ids
                text = tok.decode(ids, skip_special_tokens=True)
                if len(ids) > args.max_new - 1:
                    stats["too_long"] += 1
                elif not (reward(text, q["answer"]) and last_boxed(text) is not None):
                    stats["unverified"] += 1
                else:
                    stats[key][1] += 1
                    stats["tokens"] += len(ids) + 1
            stats["epochs_at_2.46M"] = 2.46e6 / stats["tokens"] if stats["tokens"] else None
            out[f"{name} s{seed}"] = stats
            print(f"{name} seed {seed}: c=0 {stats['c0'][1]} of {stats['c0'][0]} retained, c>0 {stats['c>0'][1]} of "
                  f"{stats['c>0'][0]}; dropped too long {stats['too_long']}, unverified {stats['unverified']}; "
                  f"{stats['tokens']} response tokens, {stats['epochs_at_2.46M']:.2f} epochs", flush=True)
    (HERE / "results" / "p19_retained.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
