"""Pre-registration 35: archives and selections for the finishing/format/content controls of the H25 start.

Sources: the base's 64 samples per training problem (Stage 0 of Pre-registration 30; texts s0-s7.json.gz, shard k at
offset 490 k) and the teacher archive data/p30_archive.json (Amendment 3). H = data/p30_sel_H.json (262 problems).
Canonicalization as in sft_start.examples: retokenized with the 1.5B's tokenizer without special tokens, at most 1,023
tokens, a final box, token-level duplicates removed; "verified" additionally means the verifier accepts it.
- One terminal format for SELF, TEACHF and FMT: each response ends right after its last box (and a following "$").
- SELF (self-SFT) and TEACHF (teacher on the same prompts): each finished (at most 1,000 retokenized tokens; the texts
  carry no finish flag), verified base response on an H problem is paired with an unused verified teacher response
  of the same problem in the same length bin (0-199, 200-399, 400-599, 600-799, 800-1,023 tokens; nearest length, ties
  by sample order); unpaired responses are dropped; F = the H problems with at least one pair. SELF and TEACHF hold
  the two sides of the same pairs.
- FMT (format-only): on the H problems, up to three finished base responses in sample order, regardless of
  correctness, that either have a final box or state an explicit final answer ("the answer is ..."), which is then
  boxed deterministically (numeric answers only; answer text preserved); trained with sft_start --verify boxed --loss-mask format (only
  tokens entirely inside the final box's markers, and the stop token, are supervised).
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
FINISHED = 1000  # completion proxy for base samples (texts carry no finish flag): at most 1,000 retokenized tokens
BINS = [0, 200, 400, 600, 800, 1024]
ANSWER_RE = re.compile(r"(?:final answer|the answer) is[:\s]*\$?(-?\d+(?:\.\d+)?(?:/\d+)?)\$?(?=[ \t]*(?:[.,;][ \t]*(?:\n|$)|\n|$))", re.I | re.M)  # a complete numeric answer: only line end or final punctuation may follow


def bin_of(L):
    return next(k for k in range(len(BINS) - 1) if BINS[k] <= L < BINS[k + 1])


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

    def canon(resp, answer, need_correct, finished=False):
        ids = enc(resp)
        text = dec(ids)
        if len(ids) > (FINISHED if finished else MAXT) or last_boxed(text) is None or (need_correct and not reward(text, answer)):
            return None
        return ids

    def terminal(resp):
        """One terminal format: the response ends right after its last box (and a closing "$" if one follows)."""
        start = resp.rfind("\\boxed{")
        depth, i = 1, start + len("\\boxed{")
        while i < len(resp) and depth:
            depth += {"{": 1, "}": -1}.get(resp[i], 0)
            i += 1
        if i < len(resp) and resp[i] == "$":
            i += 1
        return resp[:i]

    rep = {"H": len(H), "length_bins": BINS, "finished_proxy_max_tokens": FINISHED}
    self_arc, teach_arc, F = [], [], []
    no_base, no_pair = 0, 0
    for q in H:
        i, ans = qi[q], train[qi[q]]["answer"]
        seen, sols = set(), []
        for r in base[i]:
            if len(enc(r)) > FINISHED:  # completion proxy on the original sample, before cropping
                continue
            r = terminal(r) if last_boxed(r) is not None else r
            ids = canon(r, ans, True, finished=True)
            if ids is not None and tuple(ids) not in seen:
                seen.add(tuple(ids))
                sols.append((r, len(ids)))
        if not sols:
            no_base += 1
            continue
        tsols = []
        for r in teacher[q]["responses"]:
            r = terminal(r)
            ids = canon(r, ans, True)
            if ids is not None:
                tsols.append((r, len(ids)))
        pairs, used = [], set()
        for r, L in sols:  # pair each base response with an unused teacher response in the same length bin
            b = bin_of(L)
            js = [j for j in range(len(tsols)) if j not in used and bin_of(tsols[j][1]) == b]
            if js:
                j = min(js, key=lambda j: (abs(tsols[j][1] - L), j))
                used.add(j)
                pairs.append((r, tsols[j][0], L, tsols[j][1]))
        if not pairs:
            no_pair += 1
            continue
        F.append(q)
        self_arc.append({"question": q, "answer": ans, "responses": [p[0] for p in pairs]})
        teach_arc.append({"question": q, "answer": ans, "responses": [p[1] for p in pairs]})
        rep.setdefault("pair_length_differences", []).extend(p[3] - p[2] for p in pairs)
    rep["F"] = len(F)
    rep["F_excluded_no_finished_verified_base_response"] = no_base
    rep["F_excluded_no_same_bin_teacher_response"] = no_pair
    d = rep.pop("pair_length_differences")
    rep["paired_responses"] = len(d)
    rep["pair_mean_abs_length_difference"] = sum(abs(x) for x in d) / len(d)
    rep["self_responses"] = sum(len(d["responses"]) for d in self_arc)
    rep["teachF_responses"] = sum(len(d["responses"]) for d in teach_arc)
    sl = [len(enc(r)) for d in self_arc for r in d["responses"]]
    tl = [len(enc(r)) for d in teach_arc for r in d["responses"]]
    rep["self_mean_tokens"], rep["teachF_mean_tokens"] = sum(sl) / len(sl), sum(tl) / len(tl)

    fmt_arc, Hf, kinds = [], [], {"boxed": 0, "boxing_added": 0}
    for q in H:
        i, ans = qi[q], train[qi[q]]["answer"]
        seen, sols = set(), []
        for r in base[i]:
            if len(enc(r)) > FINISHED:  # completion proxy on the original sample, before cropping or boxing
                continue
            kind = "boxed"
            if last_boxed(r) is None:  # an explicit final answer without a box: box it deterministically
                m = None
                for m in ANSWER_RE.finditer(r):
                    pass
                if m is None:
                    continue
                r = r[:m.start(1)] + "\\boxed{" + m.group(1).strip() + "}" + r[m.end(1):]
                kind = "boxing_added"
            r = terminal(r)
            ids = canon(r, ans, False, finished=True)
            if ids is not None and tuple(ids) not in seen:
                seen.add(tuple(ids))
                sols.append(r)
                kinds[kind] += 1
            if len(sols) == 3:
                break
        if sols:
            Hf.append(q)
            fmt_arc.append({"question": q, "answer": ans, "responses": sols})
    rep["Hfmt"] = len(Hf)
    rep["format_kinds"] = kinds
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
