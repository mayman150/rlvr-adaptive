"""Pre-registration 36: the self-solution pool and the coverage weights of the teacher-free warm start.

Pool: every training problem with at least one finished, verified base response in Stage 0's 64 samples
(Pre-registration 30): the original sample has at most 1,000 retokenized tokens (completion proxy; the texts carry
no finish flag), its retokenized decoding verifies and has a final box; responses end right after their last box
(one terminal format, as in Pre-registration 35); token-level duplicates removed; at most 8 per problem (sample order).
Weights: the marginal pass@16 value 16 (1 - p)^15 with p the Beta-prior posterior mean from the 64-sample counts
(law.fit_prior over all 3,920 problems), as in p36_weights.py; "uniform" gives every pool problem weight 1.
Outputs (data/): p36_self_archive.json, p36_self_pool.json, p36_w_cws.json, p36_w_uniform.json, p36_prepare_report.json.
Usage: python p36_prepare.py --stage0 DIR --tokenizer-json tokenizer.json
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

import law
from verifier import last_boxed, reward

HERE = Path(__file__).resolve().parent
FINISHED, MAXT, PER = 1000, 1023, 8


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
    ap.add_argument("--stage0", default="/tmp/p30/stage0")
    ap.add_argument("--tokenizer-json", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer_json)
    enc = lambda s: tk.encode(s, add_special_tokens=False).ids
    dec = lambda ids: tk.decode(ids, skip_special_tokens=True)
    train = json.loads((HERE / "data" / "train.json").read_text())
    st = json.loads((HERE / "data" / "p30_stage0.json").read_text())
    base = {}
    for k in range(8):
        d = json.load(gzip.open(Path(a.stage0) / f"s{k}.json.gz", "rt"))
        for s in d["samples"]:
            base.setdefault(490 * k + s["problem"], []).append(s["text"])
    arc, pool, kept = [], [], []
    for i, q in enumerate(train):
        seen, sols = set(), []
        for r in base[i]:
            if len(enc(r)) > FINISHED or last_boxed(r) is None:
                continue
            r = terminal(r)
            ids = enc(r)
            text = dec(ids)
            if len(ids) > MAXT or last_boxed(text) is None or not reward(text, q["answer"]) or tuple(ids) in seen:
                continue
            seen.add(tuple(ids))
            sols.append(r)
            if len(sols) == PER:
                break
        if sols:
            arc.append({"question": q["question"], "answer": q["answer"], "responses": sols})
            pool.append(q["question"])
            kept.append(i)
    c = np.array([r["c"] for r in st])
    pa, pb = law.fit_prior(c, 64)
    p = (c + pa) / (64 + pa + pb)
    w = 16 * (1 - p) ** 15
    wc = {train[i]["question"]: float(w[i]) for i in kept}
    wu = {train[i]["question"]: 1.0 for i in kept}
    ck = c[kept]
    W = np.array([wc[train[i]["question"]] for i in kept])
    rep = {"pool_problems": len(pool), "responses": sum(len(d["responses"]) for d in arc), "prior": [pa, pb],
           "draw_shares": {lab: {"problems": int(m.sum()), "cws": float(W[m].sum() / W.sum()), "uniform": float(m.mean())}
                           for lab, m in (("c=1-4", (ck >= 1) & (ck <= 4)), ("c=5-16", (ck >= 5) & (ck <= 16)),
                                          ("c=17-32", (ck >= 17) & (ck <= 32)), ("c>32", ck > 32))}}
    out = HERE / "data"
    rep["sha256"] = {}
    for name, obj in (("p36_self_archive.json", arc), ("p36_self_pool.json", pool), ("p36_w_cws.json", wc), ("p36_w_uniform.json", wu)):
        blob = json.dumps(obj).encode()
        (out / name).write_bytes(blob)
        rep["sha256"][name] = hashlib.sha256(blob).hexdigest()
    (out / "p36_prepare_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
