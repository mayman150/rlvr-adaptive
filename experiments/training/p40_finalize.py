"""Pre-registration 40: freeze ONE common guided set and its teacher and self banks.

A problem enters G only if its teacher trace and its self trace are both audited "valid" (blind audit) and both have a
non-empty answer-safe hint prefix (hint_utils.safe_hint_lines). Inputs: banks of candidate pairs (p40_banks.py output,
one or more prefixes) and audit label files ({"labels": [{id, label}], "key": {id: {question, source}}}).
Outputs (data/): p40_guided.json, p40_bank_teacher.json, p40_bank_self.json, p40_sft_teacher.json, p40_sft_self.json,
p40_final_report.json (counts by base count, tokens, hint-prefix lengths, hashes).
Usage: python p40_finalize.py --banks p40pre [p40new] --audits results_p26/p40pre_audit_labels.json [...]
"""
import argparse
import hashlib
import json
from pathlib import Path

from hint_utils import reveals, safe_hint_lines

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", required=True)
    ap.add_argument("--audits", nargs="+", required=True)
    ap.add_argument("--max-frac", type=float, default=0.75, help="hint lines shown at the top level (as train_grpo --hint-max-frac)")
    ap.add_argument("--leak-audit", default=None, help="semantic leak audit labels: problems flagged in either source are excluded")
    a = ap.parse_args()
    lab = {}
    for f in a.audits:
        d = json.loads(Path(f).read_text())
        L = {int(x["id"]): x["label"] for x in d["labels"]}
        for k, v in d["key"].items():
            lab[(v["question"], v["source"])] = L[int(k)]
    pairs = {}
    for b in a.banks:
        for src in ("teacher", "self"):
            for x in json.loads((HERE / "data" / f"{b}_bank_{src}.json").read_text()):
                pairs.setdefault(x["question"], {})[src] = x
    train = json.loads((HERE / "data" / "train.json").read_text())
    idx = {q["question"]: i for i, q in enumerate(train)}
    c = json.loads((HERE / "data" / "p39_stage0_counts_v2.json").read_text())
    flagged = set()
    if a.leak_audit:
        flagged = {x["question"] for x in json.loads(Path(a.leak_audit).read_text())["items"] if x["leak"]}
    G, rep = [], {"excluded": {"not both audited valid": 0, "no safe prefix": 0, "leak check failed": 0, "semantic leak": 0}}
    for q in sorted(pairs, key=lambda x: idx[x]):
        p = pairs[q]
        if not all(lab.get((q, s)) == "valid" for s in ("teacher", "self")) or len(p) < 2:
            rep["excluded"]["not both audited valid"] += 1
            continue
        hint = {s: safe_hint_lines(p[s]["trace"], p[s]["answer"]) for s in ("teacher", "self")}
        if not all(hint.values()):
            rep["excluded"]["no safe prefix"] += 1
            continue
        if any(reveals(line, p[s]["answer"]) for s in ("teacher", "self") for line in hint[s]):
            rep["excluded"]["leak check failed"] += 1
            continue
        if q in flagged:
            rep["excluded"]["semantic leak"] += 1
            continue
        G.append(q)
    bt = [pairs[q]["teacher"] for q in G]
    bs = [pairs[q]["self"] for q in G]
    rep.update({"guided": len(G), "c=0": sum(c[idx[q]] == 0 for q in G), "c=1-4": sum(1 <= c[idx[q]] <= 4 for q in G),
                "teacher_tokens": sum(x["tokens"] for x in bt), "self_tokens": sum(x["tokens"] for x in bs)})
    rep["sha256"] = {}
    for name, obj in (("p40_guided.json", G), ("p40_bank_teacher.json", bt), ("p40_bank_self.json", bs),
                      ("p40_sft_teacher.json", [{"question": x["question"], "answer": x["answer"], "responses": [x["trace"]]} for x in bt]),
                      ("p40_sft_self.json", [{"question": x["question"], "answer": x["answer"], "responses": [x["trace"]]} for x in bs])):
        blob = json.dumps(obj).encode()
        (HERE / "data" / name).write_bytes(blob)
        rep["sha256"][name] = hashlib.sha256(blob).hexdigest()
    (HERE / "data" / "p40_final_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
