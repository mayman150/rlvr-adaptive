"""Audit reference answers: a model solves each problem; where its answer disagrees with the
reference (by the training verifier), a stronger model adjudicates between the two.

Usage: AWS_PROFILE=... python label_audit.py data/eval_frontier_512.json data/val_frontier_256.json
Writes <file>.audit.jsonl (cache) and prints the items whose reference the adjudicator rejects.
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import anthropic
from anthropic import AsyncAnthropicBedrock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verifier import last_boxed, reward  # noqa: E402

SOLVER = "global.anthropic.claude-haiku-5-5"
JUDGE = "global.anthropic.claude-opus-5-5"
SOLVE = """Solve the following competition math problem. Work carefully, then give only the final
answer inside \\boxed{{}} (if several values are asked for, list all of them, separated by commas).

Problem:
{question}"""
ADJUDICATE = """A competition math problem has a reference answer that may be wrong or incomplete.
Two candidate final answers are given. Solve the problem yourself, carefully and completely,
then decide.

Problem:
{question}

Candidate A (reference): {label}
Candidate B: {other}

End with the complete correct final answer inside \\boxed{{}} (all values if several are asked
for), and on the last line write exactly one of: VERDICT: A, VERDICT: B, VERDICT: NEITHER."""


async def call(client, sem, model, prompt, max_tokens):
    async with sem:
        for _ in range(3):
            try:
                r = await client.messages.create(model=model, max_tokens=max_tokens, output_config={"effort": "high"},
                                                 messages=[{"role": "user", "content": prompt}])
                return "".join(b.text for b in r.content if b.type == "text")
            except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
                err = str(e)[:200]
        return f"ERROR {err}"


async def audit(path, client, sem):
    items = json.loads(Path(path).read_text())
    cache_path = Path(str(path) + ".audit.jsonl")
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["question"]] = rec
    todo = [q for q in items if q["question"] not in cache]
    solved = await asyncio.gather(*(call(client, sem, SOLVER, SOLVE.format(question=q["question"]), 16000) for q in todo))
    recs = []
    for q, text in zip(todo, solved):
        agrees = bool(reward(text, q["answer"]))
        recs.append({"question": q["question"], "label": q["answer"], "solver_answer": last_boxed(text), "agrees": agrees})
    disagree = [r for r in recs if not r["agrees"] and r["solver_answer"]]
    judged = await asyncio.gather(*(call(client, sem, JUDGE, ADJUDICATE.format(question=r["question"], label=r["label"],
                                                                               other=r["solver_answer"]), 20000) for r in disagree))
    for r, text in zip(disagree, judged):
        m = re.findall(r"VERDICT:\s*(A|B|NEITHER)", text)
        r["verdict"] = m[-1] if m else None
        r["judge_answer"] = last_boxed(text)
        r["judge_agrees_with_label"] = bool(reward(text, r["label"]))
    with cache_path.open("a") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
            cache[r["question"]] = r
    rejected = [r for r in cache.values() if r.get("verdict") in ("B", "NEITHER") and not r.get("judge_agrees_with_label")]
    print(f"{path}: {len(items)} problems, solver agrees on {sum(r['agrees'] for r in cache.values())}, "
          f"adjudicated {sum('verdict' in r for r in cache.values())}, reference rejected on {len(rejected)}")
    for r in rejected:
        print(f"   REJECTED label {r['label']!r} -> judge {r['judge_answer']!r} ({r['verdict']}): {r['question'][:140]!r}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    client = AsyncAnthropicBedrock(aws_region=args.region, max_retries=6)
    sem = asyncio.Semaphore(args.concurrency)
    for f in args.files:
        await audit(f, client, sem)


if __name__ == "__main__":
    asyncio.run(main())
