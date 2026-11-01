"""Pre-registration 40 (pilot): the guided set and the matched teacher and self trace banks.

Guided set G: training problems with at most 4 of 64 base successes (Stage 0 census, verifier v2) that have
- a verified, finished teacher solution (Qwen2.5-Math-7B-Instruct: the archive, data/p30_archive.json, or Stage 0b's
  64-sample deep run), and
- a verified, finished base solution: for c = 1-4 from the Stage 0 census (64 samples, 1,024 tokens); for c = 0 from
  the Stage 0 harvest (1,024 samples, 2,048 tokens), restricted to problems whose audited solution is "valid" (Gate H).
For each problem in G, one teacher trace and one self trace are chosen as the pair with the smallest absolute token
difference (ties: the shorter pair), both at most --max-tokens retokenized tokens and ending at their last box.
Outputs (data/): p40_guided.json (questions), p40_bank_teacher.json and p40_bank_self.json ({question, answer,
trace}), p40_sft_teacher.json and p40_sft_self.json ({question, answer, responses: [trace]}), p40_banks_report.json.
Usage: python p40_banks.py --stage0 DIR --harvest RESULTSDIR --teacher-deep TEXTS.json.gz --audit AUDIT.json --tokenizer-json T
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from verifier import last_boxed, reward

HERE = Path(__file__).resolve().parent


def terminal(resp):
    start = resp.rfind("\\boxed{")
    depth, i = 1, start + len("\\boxed{")
    while i < len(resp) and depth:
        depth += {"{": 1, "}": -1}.get(resp[i], 0)
        i += 1
    if i < len(resp) and resp[i] == "$":
        i += 1
    return resp[:i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", required=True)
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--teacher-deep", default=None, help="Stage 0b teacher texts; omitted for the preliminary bank")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--tokenizer-json", required=True)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--prefix", default="p40", help="output file prefix (p40 for the final banks)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer_json)
    ntok = lambda s: len(tk.encode(s, add_special_tokens=False).ids)
    train = json.loads((HERE / "data" / "train.json").read_text())
    idx = {q["question"]: i for i, q in enumerate(train)}
    c = json.loads((HERE / "data" / "p39_stage0_counts_v2.json").read_text())

    def keep(text, ans, finished=True):
        if not finished or last_boxed(text) is None or not reward(text, ans):
            return None
        t = terminal(text)
        n = ntok(t)
        return (t, n) if n <= a.max_tokens and reward(t, ans) else None

    teacher = {}
    for d in json.loads((HERE / "data" / "p30_archive.json").read_text()):
        if d["question"] in idx:
            for r in d.get("responses") or []:
                k = keep(r, d["answer"])
                if k:
                    teacher.setdefault(d["question"], []).append(k)
    deep = json.load(gzip.open(a.teacher_deep, "rt")) if a.teacher_deep else {"samples": [], "problems": []}
    for s in deep["samples"]:
        q = deep["problems"][s["problem"]]
        k = keep(s["text"], q["answer"], s.get("finished", True))
        if k:
            teacher.setdefault(q["question"], []).append(k)
    selfs = {}
    for kk in range(8):
        d = json.load(gzip.open(Path(a.stage0) / f"s{kk}.json.gz", "rt"))
        for s in d["samples"]:
            i = 490 * kk + s["problem"]
            if 1 <= c[i] <= 4 and len(selfs.get(train[i]["question"], [])) < 16:
                k = keep(s["text"], train[i]["answer"])
                if k and ntok(s["text"]) <= 1000:  # finished proxy for the 1,024-token census
                    selfs.setdefault(train[i]["question"], []).append(k)
    valid = {int(x["index"]) for x in json.loads(Path(a.audit).read_text())["labels"] if x["label"] == "valid"}  # harvest audit (Gate H)
    c0 = json.loads((HERE / "data" / "p39_train_c0.json").read_text())
    for kk in range(8):
        d = json.load(gzip.open(Path(a.harvest) / f"HARVEST_c0_{kk}" / "eval" / "step0_T1.00_texts.json.gz", "rt"))
        for s in d["samples"]:
            j = 14 * kk + s["problem"]
            if j in valid and len(selfs.get(c0[j]["question"], [])) < 16:
                k = keep(s["text"], c0[j]["answer"], s.get("finished", True))
                if k:
                    selfs.setdefault(c0[j]["question"], []).append(k)
    guided, bt, bs, rep = [], [], [], {"by_c": {}}
    for q in sorted(set(teacher) & set(selfs), key=lambda x: idx[x]):
        i = idx[q]
        if c[i] > 4:
            continue
        best = min(((abs(t[1] - s[1]), t[1] + s[1], t, s) for t in teacher[q] for s in selfs[q]), key=lambda x: (x[0], x[1]))
        _, _, t, s = best
        guided.append(q)
        bt.append({"question": q, "answer": train[i]["answer"], "trace": t[0], "tokens": t[1]})
        bs.append({"question": q, "answer": train[i]["answer"], "trace": s[0], "tokens": s[1]})
        key = "c=0" if c[i] == 0 else "c=1-4"
        rep["by_c"][key] = rep["by_c"].get(key, 0) + 1
    rep.update({"guided": len(guided), "teacher_tokens": sum(x["tokens"] for x in bt), "self_tokens": sum(x["tokens"] for x in bs),
                "mean_abs_pair_diff": sum(abs(x["tokens"] - y["tokens"]) for x, y in zip(bt, bs)) / max(len(bt), 1),
                "hard_with_teacher": sum(1 for q in teacher if c[idx[q]] <= 4), "hard_with_self": sum(1 for q in selfs if c[idx[q]] <= 4)})
    out = HERE / "data"
    rep["sha256"] = {}
    for name, obj in ((f"{a.prefix}_guided.json", guided), (f"{a.prefix}_bank_teacher.json", bt), (f"{a.prefix}_bank_self.json", bs),
                      (f"{a.prefix}_sft_teacher.json", [{"question": x["question"], "answer": x["answer"], "responses": [x["trace"]]} for x in bt]),
                      (f"{a.prefix}_sft_self.json", [{"question": x["question"], "answer": x["answer"], "responses": [x["trace"]]} for x in bs])):
        blob = json.dumps(obj).encode()
        (out / name).write_bytes(blob)
        rep["sha256"][name] = hashlib.sha256(blob).hexdigest()
    (out / f"{a.prefix}_banks_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
