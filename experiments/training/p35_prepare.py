"""Pre-registration 35: archives and selections for the finishing/format/content controls of the H25 start.

Sources: the base's 64 samples per training problem (Stage 0 of Pre-registration 30; texts s0-s7.json.gz, shard k at
offset 490 k) and the teacher archive data/p30_archive.json (Amendment 3). H = data/p30_sel_H.json (262 problems).
Canonicalization as in sft_start.examples: retokenized with the 1.5B's tokenizer without special tokens, at most 1,023
tokens, a final box, token-level duplicates removed; "verified" additionally means the verifier accepts it.
- SELF (self-SFT): on F = the H problems with at least one verified base response, all verified base responses.
- TEACHF (teacher on the same feasible subset): for each F problem, as many teacher responses as it has SELF responses
  (at most the teacher's count), chosen greedily by nearest token length to the SELF responses (ties: sample order).
- FMT (format-only): on the H problems with at least one boxed base response (any correctness), the first three such
  responses in sample order; trained with sft_start --verify boxed --loss-mask format (only the final box's markers and
  the stop token are supervised).
- CORR (corrupted answer key): teacher responses on H with the content of the last box replaced by a wrong answer
  (a number n becomes n + 1; any other answer gets "+1" appended), kept only if the verifier rejects the result;
  trained with --verify boxed.
Outputs (data/): p35_sel_F.json, p35_sel_Hfmt.json, p35_archive_{self,teachF,format,corr}.json, p35_prepare_report.json.
Usage: python p35_prepare.py --stage0 DIR --tokenizer-json tokenizer.json
"""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

from verifier import as_number, last_boxed, normalize, reward

HERE = Path(__file__).resolve().parent
MAXT = 1023


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", default="/tmp/p30/stage0")
    ap.add_argument("--tokenizer-json", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer_json)
    enc = lambda s: tk.encode(s, add_special_tokens=False).ids
    dec = lambda ids: tk.decode(ids, skip_special_tokens=True)
    train = json.loads((HERE / "data" / "train.json").read_text())
    qi = {q["question"]: i for i, q in enumerate(train)}
    H = json.loads((HERE / "data" / "p30_sel_H.json").read_text())
    teacher = {d["question"]: d for d in json.loads((HERE / "data" / "p30_archive.json").read_text())}
    base = {}
    for k in range(8):
        d = json.load(gzip.open(Path(a.stage0) / f"s{k}.json.gz", "rt"))
        for s in d["samples"]:
            base.setdefault(490 * k + s["problem"], []).append(s["text"])

    def canon(resp, answer, need_correct):
        ids = enc(resp)
        text = dec(ids)
        if len(ids) > MAXT or last_boxed(text) is None or (need_correct and not reward(text, answer)):
            return None
        return ids

    rep = {"H": len(H)}
    self_arc, teach_arc, F = [], [], []
    for q in H:
        i, ans = qi[q], train[qi[q]]["answer"]
        seen, sols = set(), []
        for r in base[i]:
            ids = canon(r, ans, True)
            if ids is not None and tuple(ids) not in seen:
                seen.add(tuple(ids))
                sols.append((r, len(ids)))
        if not sols:
            continue
        F.append(q)
        self_arc.append({"question": q, "answer": ans, "responses": [r for r, _ in sols]})
        tsols = []
        for r in teacher[q]["responses"]:
            ids = canon(r, ans, True)
            if ids is not None:
                tsols.append((r, len(ids)))
        chosen, pool = [], list(range(len(tsols)))
        for _, L in sols[:len(tsols)]:
            j = min(pool, key=lambda j: (abs(tsols[j][1] - L), j))
            pool.remove(j)
            chosen.append(tsols[j][0])
        teach_arc.append({"question": q, "answer": ans, "responses": chosen})
    rep["F"] = len(F)
    rep["F_excluded_no_verified_base_response"] = len(H) - len(F)
    rep["self_responses"] = sum(len(d["responses"]) for d in self_arc)
    rep["teachF_responses"] = sum(len(d["responses"]) for d in teach_arc)
    sl = [len(enc(r)) for d in self_arc for r in d["responses"]]
    tl = [len(enc(r)) for d in teach_arc for r in d["responses"]]
    rep["self_mean_tokens"], rep["teachF_mean_tokens"] = sum(sl) / len(sl), sum(tl) / len(tl)

    fmt_arc, Hf = [], []
    for q in H:
        i, ans = qi[q], train[qi[q]]["answer"]
        seen, sols = set(), []
        for r in base[i]:
            ids = canon(r, ans, False)
            if ids is not None and tuple(ids) not in seen:
                seen.add(tuple(ids))
                sols.append(r)
            if len(sols) == 3:
                break
        if sols:
            Hf.append(q)
            fmt_arc.append({"question": q, "answer": ans, "responses": sols})
    rep["Hfmt"] = len(Hf)
    rep["format_responses"] = sum(len(d["responses"]) for d in fmt_arc)
    rep["format_correct_share"] = sum(bool(reward(r, d["answer"])) for d in fmt_arc for r in d["responses"]) / rep["format_responses"]

    def corrupt(resp):
        s = resp.rfind("\\boxed{")
        b = last_boxed(resp)
        if s < 0 or b is None:
            return None
        v = as_number(normalize(b))
        new = (str(int(v) + 1) if v is not None and v == int(v) else f"{b}+1")
        e = s + len("\\boxed{") + len(b)
        return resp[:s + len("\\boxed{")] + new + resp[e:]

    corr_arc, kept = [], 0
    for q in H:
        ans = train[qi[q]]["answer"]
        sols = []
        for r in teacher[q]["responses"]:
            c = corrupt(r)
            if c is not None and canon(c, ans, False) is not None and not reward(c, ans):
                sols.append(c)
        corr_arc.append({"question": q, "answer": ans, "responses": sols})
        kept += bool(sols)
    rep["corr_problems_with_solution"] = kept
    out = HERE / "data"
    files = {"p35_sel_F.json": F, "p35_sel_Hfmt.json": Hf, "p35_archive_self.json": self_arc,
             "p35_archive_teachF.json": teach_arc, "p35_archive_format.json": fmt_arc, "p35_archive_corr.json": corr_arc}
    rep["sha256"] = {}
    for name, obj in files.items():
        blob = json.dumps(obj).encode()
        (out / name).write_bytes(blob)
        rep["sha256"][name] = hashlib.sha256(blob).hexdigest()
    (out / "p35_prepare_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
