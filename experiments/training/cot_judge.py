"""CoT-Pass@K (Wen et al. 2025): pass@k counting only answers whose final answer AND reasoning
are correct, with Claude Haiku 5.5 on Amazon Bedrock as the reasoning judge.

Usage: AWS_PROFILE=stop-admin python cot_judge.py <run>_passk_texts_step<N>.json.gz [...]
       [--judgments 3] [--concurrency 16] [--limit-problems N]

Only answers with a correct final answer are judged (an answer with a wrong final answer can
never count). Each is judged `--judgments` times independently; D per problem is computed
under the any / majority / all aggregation rules. Judgments are cached in a .judged.jsonl file
next to the input, so reruns only pay for new answers.
"""

import argparse
import asyncio
import gzip
import json
import math
import re
from pathlib import Path

import anthropic
from anthropic import AsyncAnthropicBedrock

MODEL = "global.anthropic.claude-haiku-5-5"
PROMPT = """You are a mathematics expert grading a student's solution.

Problem:
{question}

Reference final answer: {answer}

Student solution (its final answer matches the reference):
{solution}

Judge whether the reasoning is correct, not just the final answer. Check every computation,
the logical validity of each step, the concepts used, and whether an essential step is missing.
Presentation issues do not matter. A correct final answer reached through invalid reasoning,
guessing, unjustified leaps, or contradictory steps is NOT correct.

Briefly list any errors you find (calculation, logical, conceptual, omission, other), then end
with your verdict: \\boxed{{yes}} if the reasoning is correct, \\boxed{{no}} otherwise."""


def verdict(text):
    found = re.findall(r"\\boxed\{\s*(yes|no)\s*\}", text, flags=re.IGNORECASE)
    return found[-1].lower() == "yes" if found else None


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


async def judge_one(client, sem, question, answer, solution):
    async with sem:
        try:
            r = await client.messages.create(
                model=MODEL, max_tokens=8000, output_config={"effort": "medium"},
                messages=[{"role": "user", "content": PROMPT.format(question=question, answer=answer, solution=solution)}])
        except anthropic.BadRequestError as e:
            return {"verdict": None, "error": f"bad_request: {str(e)[:200]}"}
        except anthropic.APIStatusError as e:
            return {"verdict": None, "error": f"status {e.status_code}"}
        except anthropic.APIConnectionError:
            return {"verdict": None, "error": "connection"}
        if r.stop_reason == "refusal":
            return {"verdict": None, "error": "refusal"}
        text = "".join(b.text for b in r.content if b.type == "text")
        return {"verdict": verdict(text), "tokens_in": r.usage.input_tokens, "tokens_out": r.usage.output_tokens}


async def judge_file(path, args, client):
    data = json.load(gzip.open(path, "rt"))
    problems, samples, n = data["problems"], data["samples"], data["n"]
    if args.limit_problems:
        samples = [s for s in samples if s["problem"] < args.limit_problems]
        problems = problems[: args.limit_problems]
    cache_path = Path(str(path).replace(".json.gz", ".judged.jsonl"))
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            cache[(rec["sample"], rec["j"])] = rec
    sem = asyncio.Semaphore(args.concurrency)
    todo = []
    for idx, s in enumerate(samples):
        if not s["reward"]:
            continue
        q = problems[s["problem"]]
        for j in range(args.judgments):
            if (idx, j) not in cache or cache[(idx, j)].get("verdict") is None:
                todo.append((idx, j, judge_one(client, sem, q["question"], q["answer"], s["text"])))
    if todo:
        results = await asyncio.gather(*(t[2] for t in todo))
        with cache_path.open("a") as fh:
            for (idx, j, _), res in zip(todo, results):
                rec = {"sample": idx, "j": j, **res}
                cache[(idx, j)] = rec
                fh.write(json.dumps(rec) + "\n")
    rules = {"any": lambda v: any(v), "majority": lambda v: sum(v) * 2 > len(v), "all": lambda v: all(v)}
    D = {rule: [0] * len(problems) for rule in rules}
    C = [0] * len(problems)
    unjudged = 0
    for idx, s in enumerate(samples):
        if not s["reward"]:
            continue
        C[s["problem"]] += 1
        votes = [cache.get((idx, j), {}).get("verdict") for j in range(args.judgments)]
        votes = [v for v in votes if v is not None]
        if not votes:
            unjudged += 1
            continue
        for rule, f in rules.items():
            D[rule][s["problem"]] += int(f(votes))
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= n]
    out = {"file": str(path), "step": data["step"], "n": n, "problems": len(problems), "unjudged": unjudged,
           "pass_at_k": {k: sum(pass_at_k(n, c, k) for c in C) / len(C) for k in ks},
           "cot_pass_at_k": {rule: {k: sum(pass_at_k(n, d, k) for d in D[rule]) / len(D[rule]) for k in ks}
                             for rule in rules}}
    tin = sum(r.get("tokens_in", 0) for r in cache.values())
    tout = sum(r.get("tokens_out", 0) for r in cache.values())
    out["judge_tokens"] = {"input": tin, "output": tout}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--judgments", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit-problems", type=int, default=None)
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()
    client = AsyncAnthropicBedrock(aws_region=args.region, max_retries=6)
    for f in args.files:
        res = await judge_file(Path(f), args, client)
        print(json.dumps(res))
        Path(str(f).replace(".json.gz", ".cot.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
