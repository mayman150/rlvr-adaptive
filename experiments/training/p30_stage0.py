"""Pre-registration 30, Stage 0 table: per training problem, from the base's 64 samples (T = 1, 1,024 tokens; job
worth-p31-flat-s0-20261028-183640, shards BASE15train64_0-7 at offsets 0, 490, ..., 3430): success count c, unfinished
count (no final box), and the most frequent normalized wrong answer with its count (ties: lexicographically smallest).
Membership flags: H (c <= 4), W (c <= 4, unfinished <= 31, top wrong count >= 16).
Usage: python p30_stage0.py --texts DIR (s0.json.gz ... s7.json.gz) --out data/p30_stage0.json
"""

import argparse
import gzip
import json
from pathlib import Path

from verifier import last_boxed, normalize

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="/tmp/p30/stage0")
    ap.add_argument("--out", default=str(HERE / "data" / "p30_stage0.json"))
    a = ap.parse_args()
    train = json.loads((HERE / "data" / "train.json").read_text())
    rows = [None] * len(train)
    for k in range(8):
        d = json.load(gzip.open(Path(a.texts) / f"s{k}.json.gz", "rt"))
        assert d["n"] == 64
        off = 490 * k
        per = {}
        for s in d["samples"]:
            per.setdefault(s["problem"], []).append(s)
        for i, q in enumerate(d["problems"]):
            g = per[i]
            assert len(g) == 64 and q["question"] == train[off + i]["question"], (k, i)
            c = sum(bool(s["reward"]) for s in g)
            unf, wrong = 0, {}
            for s in g:
                b = last_boxed(s["text"])
                if b is None:
                    unf += 1
                elif not s["reward"]:
                    w = normalize(b)
                    wrong[w] = wrong.get(w, 0) + 1
            top = min(wrong.items(), key=lambda kv: (-kv[1], kv[0])) if wrong else (None, 0)
            rows[off + i] = {"index": off + i, "question": q["question"], "subject": train[off + i]["subject"],
                             "level": train[off + i]["level"], "c": c, "n": 64, "unfinished": unf,
                             "top_wrong": top[0], "top_wrong_count": top[1], "H": c <= 4,
                             "W": c <= 4 and unf <= 31 and top[1] >= 16}
    assert all(r is not None for r in rows)
    Path(a.out).write_text(json.dumps(rows))
    H = [r for r in rows if r["H"]]
    W = [r for r in rows if r["W"]]
    print(f"{len(rows)} problems; H {len(H)} (c = 0: {sum(r['c'] == 0 for r in rows)}); W {len(W)}; "
          f"base pass@1 {sum(r['c'] for r in rows) / (64 * len(rows)):.4f}")
    for lab, S in (("H", H), ("W", W)):
        by = {}
        for r in S:
            by[r["subject"]] = by.get(r["subject"], 0) + 1
        print(f"   {lab} by subject: {by}; by bin: {{0: {sum(r['c'] == 0 for r in S)}, '1-2': {sum(1 <= r['c'] <= 2 for r in S)}, "
              f"'3-4': {sum(3 <= r['c'] <= 4 for r in S)}}}")


if __name__ == "__main__":
    main()
