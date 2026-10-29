"""Pre-registration 30 (Amendment 2): teacher archive and selections H and R.

Teacher pool: Qwen2.5-Math-7B-Instruct samples (Amendment 1; eval_vllm texts of the four 980-problem shards, offsets read
from each shard's eval fingerprint; the two 48-problem gate outputs are not used). A response survives as in
sft_start.examples (Pre-registration 14's canonicalization): retokenized without special tokens with the 1.5B's
tokenizer, at most 1,023 tokens, its decoding verifies and has a final box, and it is not a token-level duplicate.
H = Stage 0's H (c <= 4 of 64) restricted to problems with a surviving solution; N = |H|.
R = N problems drawn uniformly without replacement (random.Random(3030)) from all problems with a surviving solution,
stratified to H's subject mix (subjects in sorted order, candidates in training-index order before the draw).
Outputs (data/): p30_archive.json (every training problem: question, answer, base c and n, verified teacher responses
in sample order), p30_sel_H.json and p30_sel_R.json (question lists in training-index order), p30_select_report.json.
Usage: python p30_select.py --teacher T1.json.gz:EVAL1.json ... (--tokenizer-json tokenizer.json | --model DIR)
"""

import argparse
import collections
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verifier import last_boxed, reward  # noqa: E402

MAX_TOKENS = 1023  # max_new - 1 at 1,024 tokens


def tokenizer(a):
    if a.tokenizer_json:
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(a.tokenizer_json)
        return (lambda s: t.encode(s, add_special_tokens=False).ids), (lambda ids: t.decode(ids, skip_special_tokens=True))
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(a.model)
    return (lambda s: t(s, add_special_tokens=False).input_ids), (lambda ids: t.decode(ids, skip_special_tokens=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", nargs="+", required=True, help="TEXTS.json.gz:EVAL.json per shard")
    ap.add_argument("--stage0", default=str(HERE / "data" / "p30_stage0.json"))
    ap.add_argument("--tokenizer-json", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=3030)
    ap.add_argument("--out", default=str(HERE / "data"))
    a = ap.parse_args()
    encode, decode = tokenizer(a)
    train = json.loads((HERE / "data" / "train.json").read_text())
    st = json.loads(Path(a.stage0).read_text())
    assert [r["question"] for r in st] == [q["question"] for q in train]
    resp = [[] for _ in train]
    seen_idx, stats = set(), collections.Counter()
    sol_tokens = collections.defaultdict(list)
    for spec in a.teacher:
        tf, ef = spec.split(":")
        fp = json.loads(Path(ef).read_text())["fingerprint"]
        assert fp["problems"] == "train.json" and fp["n"] == 4 and fp["max_new"] == 2048 and fp["seed"] == 2929, fp
        d = json.load(gzip.open(tf, "rt"))
        off = fp["offset"]
        for i, q in enumerate(d["problems"]):
            assert q["question"] == train[off + i]["question"] and off + i not in seen_idx
            seen_idx.add(off + i)
        dup = collections.defaultdict(set)
        for s in d["samples"]:
            i = off + s["problem"]
            stats["samples"] += 1
            if not s["reward"]:
                stats["not_verified"] += 1
                continue
            ids = encode(s["text"])
            text = decode(ids)
            if len(ids) > MAX_TOKENS:
                stats["too_long"] += 1
            elif not (reward(text, train[i]["answer"]) and last_boxed(text) is not None):
                stats["unverified_after_retokenization"] += 1
            elif tuple(ids) in dup[i]:
                stats["duplicate"] += 1
            else:
                dup[i].add(tuple(ids))
                resp[i].append(s["text"])
                sol_tokens[i].append(len(ids))
                stats["surviving"] += 1
    assert seen_idx == set(range(len(train))), f"shards cover {len(seen_idx)} of {len(train)} problems"
    surv = [i for i in range(len(train)) if resp[i]]
    H = [i for i in surv if st[i]["H"]]
    mix = collections.Counter(st[i]["subject"] for i in H)
    rng = random.Random(a.seed)
    R, short = [], {}
    for subj in sorted(mix):
        cand = [i for i in surv if st[i]["subject"] == subj]
        if len(cand) < mix[subj]:
            short[subj] = mix[subj] - len(cand)
        R += rng.sample(cand, min(mix[subj], len(cand)))
    assert not short, f"subject shortfall {short}"  # not expected: every subject has hundreds of surviving problems
    R.sort()
    out = Path(a.out)
    archive = [{"question": train[i]["question"], "answer": train[i]["answer"], "c": st[i]["c"], "n": st[i]["n"],
                "responses": resp[i]} for i in range(len(train))]
    files = {"p30_archive.json": archive, "p30_sel_H.json": [train[i]["question"] for i in H],
             "p30_sel_R.json": [train[i]["question"] for i in R]}
    digest = {}
    for name, obj in files.items():
        blob = json.dumps(obj).encode()
        (out / name).write_bytes(blob)
        digest[name] = hashlib.sha256(blob).hexdigest()

    def describe(S):
        bins = collections.Counter("0" if st[i]["c"] == 0 else "1-2" if st[i]["c"] <= 2 else "3-4" if st[i]["c"] <= 4 else ">=5" for i in S)
        toks = [t for i in S for t in sol_tokens[i]]
        return {"problems": len(S), "subjects": dict(sorted(collections.Counter(st[i]["subject"] for i in S).items())),
                "base_count_bins": dict(sorted(bins.items())), "solutions": len(toks),
                "solutions_per_problem": round(len(toks) / len(S), 3), "mean_solution_tokens": round(sum(toks) / len(toks), 1),
                "levels": dict(sorted(collections.Counter(train[i]["level"] for i in S).items()))}
    Hall = [i for i in range(len(train)) if st[i]["H"]]
    report = {"teacher_samples": dict(stats), "problems_with_surviving_solution": len(surv),
              "H_stage0": len(Hall), "H_surviving_by_bin": {
                  b: f"{sum(1 for i in H if f(st[i]['c']))} of {sum(1 for i in Hall if f(st[i]['c']))}"
                  for b, f in (("0", lambda c: c == 0), ("1-2", lambda c: 1 <= c <= 2), ("3-4", lambda c: 3 <= c <= 4))},
              "N": len(H), "H": describe(H), "R": describe(R), "R_overlap_with_H": len(set(R) & set(H)),
              "sha256": digest, "seed": a.seed}
    (out / "p30_select_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
