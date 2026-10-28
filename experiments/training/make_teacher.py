"""Teacher archive for Pre-registration 26 (Step 2b) from eval_vllm's texts of the 7B teacher on the base-rare
training problems: every training problem listed with its base count (c, n) from the discovery file; base-rare problems
also list their verified teacher samples in sample order ("responses"), from which the trainer's loader keeps the first
that survives canonicalization. Usage: python make_teacher.py TEXTS.json.gz [...] --out data/teacher_rare.json"""
import argparse
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verifier import last_boxed, reward  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("texts", nargs="+")
ap.add_argument("--discovery", default="discovery_qwen_train32.json")
ap.add_argument("--max-c", type=int, default=4)
ap.add_argument("--out", required=True)
a = ap.parse_args()
disc = {d["question"]: d for d in json.loads((HERE / "data" / a.discovery).read_text())}
train = json.loads((HERE / "data" / "train.json").read_text())
sols, seen = {}, set()
for f in a.texts:
    with gzip.open(f, "rt") as fh:
        t = json.load(fh)
    for i, q in enumerate(t["problems"]):
        seen.add(q["question"])
        sols.setdefault(q["question"], [])
    for smp in t["samples"]:
        q = t["problems"][smp["problem"]]
        if smp["reward"] and reward(smp["text"], q["answer"]) and last_boxed(smp["text"]) is not None:
            sols[q["question"]].append(smp["text"])
rare = [q for q in train if disc[q["question"]]["c"] <= a.max_c]
missing = [q for q in rare if q["question"] not in seen]
assert not missing, f"{len(missing)} rare problems have no teacher samples"
out = []
for q in train:
    d = disc[q["question"]]
    rec = {"question": q["question"], "answer": q["answer"], "c": d["c"], "n": d["n"], "response": None}
    if d["c"] <= a.max_c and sols[q["question"]]:
        rec["responses"] = sols[q["question"]]
    out.append(rec)
Path(a.out).write_text(json.dumps(out))
k = [len(sols[q["question"]]) for q in rare]
print(json.dumps({"train": len(train), "rare": len(rare), "rare_with_teacher_solution": sum(x > 0 for x in k),
                  "mean_correct_of_8": sum(k) / len(k),
                  "by_base_c": {c: [sum(1 for q, x in zip(rare, k) if disc[q["question"]]["c"] == c and x > 0),
                                    sum(1 for q in rare if disc[q["question"]]["c"] == c)] for c in range(a.max_c + 1)}}, indent=1))
