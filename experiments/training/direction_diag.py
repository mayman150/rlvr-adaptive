"""Diagnostics for the shared-direction explanation (Pre-registration 26, step 1).

(a) Weight directions. For each run's checkpoint, the update Delta W = W_ckpt - W_base of every 2-D non-embedding weight
    matrix: rank-1 energy share sigma_1^2 / sum sigma^2 (matrix rank, AlphaRL's measure), the cosine between runs'
    flattened updates, the alignment of their top singular directions (|<u_1, u_1'>| |<v_1, v_1'>|), and, along one
    run's trajectory, the cosine between updates at different steps and the eigenvalue shares of the steps' Gram matrix
    (trajectory rank, RELEX's measure; globally over the flattened update and per matrix). Writes three checkpoints from
    the reference run's final update: W_base + rank-1 part ("par"), W_base + the remainder ("perp"), and W_base
    ("other"); all three keep the trained embeddings, norms and biases, so "other" is the control for those parameters.
    Reports how much of each intended matrix update survives bf16 rounding.
(b) Gradient correlations across problems (needs a GPU): at given checkpoints, Dr. GRPO surrogate gradients of single
    problems from two independent groups of 8 answers at T = 1 (problems mixed in both groups), compressed by the
    CountSketch of the whitening pilot (2-D non-embedding matrices only). Reports the eigenvalue shares of each group's
    Gram matrix and of the cross-group Gram matrix (S_A S_B^T symmetrized; sampling noise is independent across
    groups, so its expectation is the Gram matrix of the expected gradients), the same for the cosine matrix
    (norm-free), the share along the batch-mean direction, and the cross-group cosine of each problem's gradient.

Usage: python direction_diag.py --model BASE --ckpts DIR --runs B1k8,PSR1k8,... --ref B1k8 --steps 25,50,100,150
       --out DIR [--grad-ckpts B1k8/ckpt_step50,B1k8/ckpt_step150 --problems 32]
"""

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Ckpt:
    """Lazy tensor access to a (possibly sharded) safetensors checkpoint."""

    def __init__(self, path):
        self.handles = {}
        files = sorted(Path(path).glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors in {path}")
        for f in files:
            h = safe_open(str(f), framework="pt", device="cpu")
            for k in h.keys():
                self.handles[k] = h

    def keys(self):
        return list(self.handles)

    def shape(self, k):
        return self.handles[k].get_slice(k).get_shape()

    def get(self, k):
        return self.handles[k].get_tensor(k)


def is_matrix(name, shape):
    return len(shape) == 2 and "embed" not in name and "lm_head" not in name


def ratio(a, b):
    """a / b, or None when b is not a finite positive number."""
    return a / b if b > 0 and math.isfinite(b) and math.isfinite(a) else None


def shares(ev):
    """Top-5 eigenvalues (descending) over the sum of their absolute values, and the negative share of that sum
    (nonzero only for indefinite estimates such as the cross-group Gram matrix)."""
    ev = ev.flip(0)
    tot = float(ev.abs().sum())
    if not (tot > 0 and math.isfinite(tot)):
        return {"top5": [], "negative_mass": None}
    return {"top5": [float(x) / tot for x in ev[:5]], "negative_mass": float(-ev.clamp(max=0).sum()) / tot}


def weight_part(args, dev):
    base = Ckpt(args.model)
    keys = [k for k in base.keys() if is_matrix(k, base.shape(k))]
    runs = [r for r in args.runs.split(",") if r]
    final = {r: Ckpt(Path(args.ckpts) / r / f"ckpt_step{args.final}") for r in runs}
    steps = [int(s) for s in args.steps.split(",") if s]
    traj = {s: Ckpt(Path(args.ckpts) / args.ref / f"ckpt_step{s}") for s in steps}
    pairs = [(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
    num = {r: 0.0 for r in runs}
    den = {r: 0.0 for r in runs}
    dot = {p: 0.0 for p in pairs}
    al = {p: 0.0 for p in pairs}
    wsum = {p: 0.0 for p in pairs}
    gram = torch.zeros(len(steps), len(steps), dtype=torch.float64)
    traj_r1, traj_e = 0.0, 0.0
    ref_top = {}
    for k in keys:
        w0 = base.get(k).to(dev).float()
        tops = {}
        for r in runs:
            d = final[r].get(k).to(dev).float() - w0
            u, sv, vh = torch.linalg.svd(d, full_matrices=False)
            e = float((sv ** 2).sum())
            num[r] += float(sv[0] ** 2)
            den[r] += e
            tops[r] = (u[:, 0], vh[0], float(sv[0]), e, d)
            if r == args.ref:
                ref_top[k] = (u[:, 0].cpu(), vh[0].cpu(), float(sv[0]))
        for a, b in pairs:
            ua, va, _, ea, da = tops[a]
            ub, vb, _, eb, db = tops[b]
            dot[(a, b)] += float((da * db).sum())
            al[(a, b)] += abs(float(ua @ ub)) * abs(float(va @ vb)) * math.sqrt(ea * eb)
            wsum[(a, b)] += math.sqrt(ea * eb)
        del tops
        D = torch.stack([(traj[s].get(k).to(dev).float() - w0).reshape(-1) for s in steps])
        gk = (D @ D.T).double().cpu()
        gram += gk
        if float(gk.trace()) > 0:
            traj_r1 += float(torch.linalg.eigvalsh(gk)[-1])
            traj_e += float(gk.trace())
        del D, w0
    rep = {"matrices": len(keys), "rank1_share": {r: ratio(num[r], den[r]) for r in runs},
           "cosine_full": {f"{a}|{b}": ratio(dot[(a, b)], math.sqrt(den[a] * den[b])) for a, b in pairs},
           "top_direction_alignment": {f"{a}|{b}": ratio(al[(a, b)], wsum[(a, b)]) for a, b in pairs},
           "trajectory_cosine": {}, "trajectory_norm_ratio": {},
           "trajectory_gram_shares": shares(torch.linalg.eigvalsh(gram)),
           "trajectory_rank1_share_per_matrix": traj_r1 / traj_e if traj_e > 0 else None}
    for i, s in enumerate(steps):
        for j in range(i + 1, len(steps)):
            t = steps[j]
            rep["trajectory_cosine"][f"{s}|{t}"] = ratio(float(gram[i, j]), math.sqrt(float(gram[i, i] * gram[j, j])))
            nr = ratio(float(gram[j, j]), float(gram[i, i]))
            rep["trajectory_norm_ratio"][f"{t}/{s}"] = math.sqrt(nr) if nr is not None else None
    # rank-1 part, remainder and neither, of the reference run's final update, as loadable checkpoints
    ref_final = Ckpt(Path(args.ckpts) / args.ref / f"ckpt_step{args.final}")
    src = Path(args.ckpts) / args.ref / f"ckpt_step{args.final}"
    rep["bf16_retention"] = {}
    for name in ("par", "perp", "other"):
        sd, dt, rr, ii = {}, 0.0, 0.0, 0.0
        for k in ref_final.keys():
            w = ref_final.get(k)
            if k in ref_top:
                w0 = base.get(k).to(dev).float()
                d = w.to(dev).float() - w0
                u, v, s1 = ref_top[k]
                r1 = s1 * torch.outer(u.to(dev), v.to(dev))
                want = r1 if name == "par" else d - r1 if name == "perp" else torch.zeros_like(d)
                out = (w0 + want).to(w.dtype)
                got = out.float() - w0
                dt += float((got * want).sum())
                rr += float((got * got).sum())
                ii += float((want * want).sum())
                sd[k] = out.cpu()
                del w0, d, r1, want, got
            else:  # embeddings, norms, biases: trained values in all three
                sd[k] = w.clone()
        if name != "other":
            nr = ratio(rr, ii)  # aggregate over matrices
            rep["bf16_retention"][name] = {"cosine": ratio(dt, math.sqrt(rr * ii)), "norm_ratio": math.sqrt(nr) if nr is not None else None}
        out = Path(args.out) / f"{args.ref}_{name}" / f"ckpt_step{args.final}"
        out.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in sd.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
        for f in src.iterdir():
            if f.suffix in (".json", ".jinja") and f.name != "model.safetensors.index.json":
                (out / f.name).write_bytes(f.read_bytes())
        del sd
        gc.collect()
    return rep


def gram_report(S):
    S = S.double()
    nrm = S.norm(dim=1, keepdim=True).clamp(min=1e-300)
    C = (S / nrm) @ (S / nrm).T
    G = S @ S.T
    off = C[~torch.eye(len(S), dtype=bool, device=C.device)]
    mean_dir = S.mean(0)
    md = float(mean_dir @ mean_dir)
    return {"gram_shares": shares(torch.linalg.eigvalsh(G)), "cosine_shares": shares(torch.linalg.eigvalsh(C)),
            "mean_direction_share": ratio(float(((S @ mean_dir) ** 2).sum()) / md, float(G.trace())) if md > 0 else None,
            "mean_pairwise_cosine": float(off.mean()),
            "pairwise_cosine_quartiles": [float(x) for x in torch.quantile(off, torch.tensor([0.25, 0.5, 0.75], dtype=off.dtype, device=off.device))]}


def grad_part(args, dev):
    from train_grpo import Runner, build_prompt, fixed_advantage, problem_sketch_tables, sketch_grads
    from verifier import reward
    problems = json.loads((HERE / "data" / "train.json").read_text())
    rng = random.Random(4242)
    rng.shuffle(problems)
    rep = {}
    for ck in [c for c in args.grad_ckpts.split(",") if c]:
        a = argparse.Namespace(model=str(Path(args.ckpts) / ck), device="cuda", gen_batch=64, prefill_tokens=256,
                               max_new=args.max_new, train_batch=4)
        runner = Runner(a)
        tables = problem_sketch_tables(runner, args.sketch_dim)
        packs, kept, tried = ([], []), [], 0
        for j, q in enumerate(problems):
            if len(kept) >= args.problems or tried >= args.problems * 4:
                break
            tried += 1
            text = build_prompt(runner.tok, q["question"])
            ans16 = runner.generate([text], 16, args.max_new, 7000 + j)  # two independent groups of 8
            groups = [ans16[:8], ans16[8:]]
            rs = [[reward(runner.tok.decode(x), q["answer"]) for _, x in ans] for ans in groups]
            if not all(0 < sum(r) < 8 for r in rs):
                continue
            for g, (ans, r) in enumerate(zip(groups, rs)):
                runner.model.train()  # generation leaves the model in evaluation mode (no checkpointing)
                runner.model.zero_grad(set_to_none=True)
                m = sum(r) / 8
                w = torch.tensor([fixed_advantage("baseline", x, m, 1.0, 0.1) for x in r], device=dev)
                for s in range(0, 8, a.train_batch):
                    lp = runner.logprob_sum([(p.to(dev), x.to(dev)) for p, x in ans[s:s + a.train_batch]])
                    (-(w[s:s + a.train_batch] * lp).sum()).backward()
                    del lp  # its graph would keep the model alive after the checkpoint is released
                packs[g].append(sketch_grads(tables, args.sketch_dim))
            kept.append((j, sum(rs[0]), sum(rs[1])))
        runner.model.zero_grad(set_to_none=True)
        del tables, runner
        gc.collect()
        torch.cuda.empty_cache()
        out = {"problems": len(kept), "tried": tried, "success_counts": [[c0, c1] for _, c0, c1 in kept]}
        if len(kept) >= 2:
            A, B = torch.stack(packs[0]).double(), torch.stack(packs[1]).double()
            out["group_a"], out["group_b"] = gram_report(A), gram_report(B)
            X = A @ B.T
            X = (X + X.T) / 2
            out["cross_gram_shares"] = shares(torch.linalg.eigvalsh(X))
            na, nb = A.norm(dim=1).clamp(min=1e-300), B.norm(dim=1).clamp(min=1e-300)
            same = (A * B).sum(1) / (na * nb)
            Xc = (A / na[:, None]) @ (B / nb[:, None]).T
            Xc = (Xc + Xc.T) / 2
            out["same_problem_cross_cosine"] = {"mean": float(same.mean()), "quartiles": [float(x) for x in torch.quantile(same, torch.tensor([0.25, 0.5, 0.75], dtype=same.dtype, device=same.device))]}
            out["cross_cosine_shares"] = shares(torch.linalg.eigvalsh(Xc))
            out["zero_sketches"] = int((A.norm(dim=1) == 0).sum() + (B.norm(dim=1) == 0).sum())
        else:
            out["note"] = "fewer than two problems mixed in both groups"
        rep[ck] = out
        del packs
        torch.cuda.empty_cache()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--runs", default="")
    ap.add_argument("--ref", default="B1k8")
    ap.add_argument("--steps", default="50,100,150")
    ap.add_argument("--final", type=int, default=150)
    ap.add_argument("--grad-ckpts", default="")
    ap.add_argument("--problems", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--sketch-dim", type=int, default=1 << 20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rep = {}
    if args.runs:
        rep["weights"] = weight_part(args, dev)
        print(json.dumps(rep["weights"], indent=1), flush=True)
        (Path(args.out) / "direction_diag.json").write_text(json.dumps(rep, indent=1))
    if args.grad_ckpts:
        rep["gradients"] = grad_part(args, dev)
        print(json.dumps(rep["gradients"], indent=1), flush=True)
    (Path(args.out) / "direction_diag.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
