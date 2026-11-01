"""Pre-registration 39: the frozen hard validation and test sets (OlympiadBench + AIME), rebuilt after Codex's review.

Sources (downloaded to their own directory, read as data only):
- OlympiadBench OE_TO_maths_en_COMP (Hugging Face Hothan/OlympiadBench, parquet): kept if single-answer, no unit, the
  answer is a rational number under verifier.normalize/as_number, and the question is standalone (no relay inputs).
- AIME 1983-2024 (Hugging Face di-zhang-fdu/AIME_1983_2024, csv): integer answers; dropped if it refers to a missing
  picture (".png", "shown", "figure", "diagram", "picture") without [asy] code.
Deduplication: word tokens [a-z0-9]+ of the lowercased question; 8-grams; overlap share = |G(q) & G(ref)| / |G(q)|.
A problem is removed if its share against any problem of MATH train (data/train_all_subjects.json, data/train.json)
or the 511 test problems is >= 0.30; pool problems with share >= 0.30 against each other form clusters, of which
the first (sorted by id) is kept. Split: seed 39, stratified by source, 100 + 100 validation, 200 + 200 test.
Usage: python -I build_hard_sets.py --olympiad PARQUET --aime CSV   (writes data/hard_val.json, data/hard_test.json,
data/hard_sets_report.json)
"""
import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

from verifier import as_number, normalize, reward

HERE = Path(__file__).resolve().parent
RELAY = re.compile(r"you will receive|TNYWR|the answer to (the )?(previous|problem)", re.I)
PICTURE = re.compile(r"\.png|as shown|shown below|shown above|figure|diagram|picture", re.I)


def grams(text, n=8):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olympiad", required=True)
    ap.add_argument("--aime", required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    pool, dropped = [], {}

    def drop(why):
        dropped[why] = dropped.get(why, 0) + 1

    for r in pq.read_table(a.olympiad).to_pylist():
        fa = r["final_answer"] or []
        q = ((r.get("context") or "") + "\n" + r["question"]).strip() if r.get("context") else r["question"]
        if r["is_multiple_answer"] or r.get("unit") or len(fa) != 1:
            drop("OB: multiple answers or unit")
        elif as_number(normalize(fa[0])) is None or not reward(f"\\boxed{{{fa[0]}}}", fa[0]):
            drop("OB: answer not rational")
        elif RELAY.search(q):
            drop("OB: relay input")
        else:
            pool.append({"question": q, "answer": fa[0].strip().strip("$"), "level": "OlympiadBench", "subject": r["subfield"], "id": f"OB-{r['id']}"})
    for r in csv.DictReader(open(a.aime)):
        ans, q = str(r["Answer"]).strip(), r["Question"].strip()
        if as_number(normalize(ans)) is None:
            drop("AIME: answer not rational")
        elif PICTURE.search(q) and "[asy]" not in q:
            drop("AIME: missing picture")
        else:
            pool.append({"question": q, "answer": ans, "level": "AIME", "subject": "AIME", "id": f"AIME-{r['Year']}-{r.get('Part') or ''}-{r['Problem Number']}"})
    ref = set()
    for f in ("train_all_subjects.json", "train.json", "eval_frontier_512.json"):
        for q in json.loads((HERE / "data" / f).read_text()):
            ref |= grams(q["question"])
    clean = []
    for q in pool:
        G = grams(q["question"])
        if G and len(G & ref) / len(G) >= 0.30:
            drop(f"{q['level']}: overlaps MATH")
        else:
            clean.append(q)
    clean.sort(key=lambda q: q["id"])
    kept, seen = [], []
    for q in clean:
        G = grams(q["question"])
        if any(G and len(G & H) / len(G) >= 0.30 for H in seen):
            drop(f"{q['level']}: duplicate within pool")
            continue
        seen.append(G)
        kept.append(q)
    rng = random.Random(39)
    val, test = [], []
    for src in ("OlympiadBench", "AIME"):
        L = [q for q in kept if q["level"] == src]
        rng.shuffle(L)
        val += L[:100]
        test += L[100:300]
    rep = {"builder": "build_hard_sets.py", "pool_after_filters": {s: sum(q["level"] == s for q in kept) for s in ("OlympiadBench", "AIME")},
           "dropped": dropped, "val": len(val), "test": len(test)}
    for name, obj in (("hard_val.json", val), ("hard_test.json", test)):
        blob = json.dumps(obj, indent=0).encode()
        (HERE / "data" / name).write_bytes(blob)
        rep[name] = hashlib.sha256(blob).hexdigest()
    (HERE / "data" / "hard_sets_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
