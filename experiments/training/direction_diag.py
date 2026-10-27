"""Diagnostics for the shared-direction explanation (Pre-registration 26, step 1).

(a) Weight directions. For each run's checkpoint, the update Delta W = W_ckpt - W_base of every 2-D weight matrix:
    rank-1 energy share sigma_1^2 / sum sigma^2, the cosine between runs' flattened updates, the alignment of their top
    singular directions (|<u_1, u_1'>| |<v_1, v_1'>|), and, along one run's trajectory, the cosine between updates at
    different steps (linearity). Writes two checkpoints built from the reference run's final update: W_base + rank-1
    part ("par") and W_base + the remainder ("perp"), for evaluation.
(b) Gradient correlations across problems (needs a GPU): at given checkpoints, Dr. GRPO surrogate gradients of single
    problems (fresh groups of 8 answers at T = 1; mixed groups only), compressed by a fixed CountSketch to D dimensions;
    their Gram matrix's eigenvalue shares, and the share captured by the batch-mean direction.

Usage: python direction_diag.py --model BASE --ckpts DIR --runs B1k8,PSR1k8,... --ref B1k8 --steps 50,100,150
       --out DIR [--grad-ckpts B1k8/ckpt_step50,B1k8/ckpt_step100 --problems 32]
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(path):
    path = Path(path)
    files = sorted(path.glob("*.safetensors"))
    out = {}
    for f in files:
        out.update(load_file(str(f)))
    return out


def matrices(sd):
    return {k: v for k, v in sd.items() if v.ndim == 2 and "embed" not in k and "lm_head" not in k}


def weight_part(args, dev):
    base = matrices(load(args.model))
    runs = [r for r in args.runs.split(",") if r]
    final = {r: matrices(load(Path(args.ckpts) / r / f"ckpt_step{args.final}")) for r in runs}
    rep = {"rank1_share": {}, "cosine_full": {}, "top_direction_alignment": {}, "trajectory_cosine": {}}
    tops = {}
    for r in runs:
        num, den, tops[r] = 0.0, 0.0, {}
        for k, w0 in base.items():
            d = (final[r][k].float() - w0.float()).to(dev)
            s = torch.linalg.svdvals(d)
            num += float(s[0] ** 2)
            den += float((s ** 2).sum())
            u, sv, vh = torch.linalg.svd(d, full_matrices=False)
            tops[r][k] = (u[:, 0].cpu(), vh[0].cpu(), float(sv[0]), float((sv ** 2).sum()))
        rep["rank1_share"][r] = num / den  # energy-weighted over matrices
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            dot = na = nb = 0.0
            al, wsum = 0.0, 0.0
            for k, w0 in base.items():
                da, db = final[a][k].float() - w0.float(), final[b][k].float() - w0.float()
                dot += float((da * db).sum())
                na += float((da * da).sum())
                nb += float((db * db).sum())
                ua, va, sa, ea = tops[a][k]
                ub, vb, sb, eb = tops[b][k]
                al += abs(float(ua @ ub)) * abs(float(va @ vb)) * math.sqrt(ea * eb)
                wsum += math.sqrt(ea * eb)
            rep["cosine_full"][f"{a}|{b}"] = dot / math.sqrt(na * nb)
            rep["top_direction_alignment"][f"{a}|{b}"] = al / wsum
    steps = [int(s) for s in args.steps.split(",") if s]
    traj = {s: matrices(load(Path(args.ckpts) / args.ref / f"ckpt_step{s}")) for s in steps}
    for i, s in enumerate(steps):
        for t in steps[i + 1:]:
            dot = na = nb = 0.0
            for k, w0 in base.items():
                da, db = traj[s][k].float() - w0.float(), traj[t][k].float() - w0.float()
                dot += float((da * db).sum())
                na += float((da * da).sum())
                nb += float((db * db).sum())
            rep["trajectory_cosine"][f"{s}|{t}"] = dot / math.sqrt(na * nb)
            rep.setdefault("trajectory_norm_ratio", {})[f"{t}/{s}"] = math.sqrt(nb / na)
    # rank-1 part and remainder of the reference run's final update, as loadable checkpoints
    full_base = load(args.model)
    ref_final = load(Path(args.ckpts) / args.ref / f"ckpt_step{args.final}")
    for name in ("par", "perp"):
        sd = {}
        for k, w0 in full_base.items():
            w = ref_final[k]
            if k in base:
                u, v, s1, _ = tops[args.ref][k]
                r1 = s1 * torch.outer(u, v)
                d = w.float() - w0.float()
                sd[k] = (w0.float() + (r1 if name == "par" else d - r1)).to(w.dtype)
            else:  # embeddings, norms, biases: trained values in both, so the pair isolates the matrices' rank-1 part
                sd[k] = w.clone()
        out = Path(args.out) / f"{args.ref}_{name}" / f"ckpt_step{args.final}"
        out.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in sd.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
        src = Path(args.ckpts) / args.ref / f"ckpt_step{args.final}"
        for f in src.iterdir():
            if f.suffix in (".json", ".jinja") and f.name != "model.safetensors.index.json":
                (out / f.name).write_bytes(f.read_bytes())
    return rep


def sketch(grad_list, dim, seed=0):
    """CountSketch of the concatenated gradient: fixed bucket and sign per parameter (regenerated per tensor)."""
    out = torch.zeros(dim, device=grad_list[0].device)
    for i, g in enumerate(grad_list):
        gen = torch.Generator(device=g.device).manual_seed(seed * 100003 + i)
        flat = g.reshape(-1).float()
        idx = torch.randint(0, dim, (flat.numel(),), generator=gen, device=g.device)
        sgn = torch.randint(0, 2, (flat.numel(),), generator=gen, device=g.device).float() * 2 - 1
        out.index_add_(0, idx, flat * sgn)
    return out


def grad_part(args, dev):
    from train_grpo import Runner, build_prompt, fixed_advantage
    from verifier import reward
    problems = json.loads((HERE / "data" / "train.json").read_text())
    rng = random.Random(4242)
    rng.shuffle(problems)
    rep = {}
    for ck in [c for c in args.grad_ckpts.split(",") if c]:
        a = argparse.Namespace(model=str(Path(args.ckpts) / ck), device="cuda", gen_batch=64, prefill_tokens=256,
                               max_new=args.max_new, train_batch=4)
        runner = Runner(a)
        runner.model.train()
        params = [p for p in runner.params]
        sk, kept = [], []
        for j, q in enumerate(problems[: args.problems * 3]):
            if len(sk) >= args.problems:
                break
            text = build_prompt(runner.tok, q["question"])
            ans = runner.generate([text], 8, args.max_new, 7000 + j)
            r = [reward(runner.tok.decode(x), q["answer"]) for _, x in ans]
            if not 0 < sum(r) < 8:
                continue
            m = sum(r) / 8
            runner.model.zero_grad(set_to_none=True)
            lp = runner.logprob_sum([(p.to(dev), x.to(dev)) for p, x in ans])
            w = torch.tensor([fixed_advantage("baseline", x, m, 1.0, 0.1) for x in r], device=dev)
            (-(w * lp).sum()).backward()
            sk.append(sketch([p.grad for p in params], args.sketch_dim))
            kept.append((j, sum(r)))
        runner.model.zero_grad(set_to_none=True)
        S = torch.stack(sk)  # problems x D
        G = S @ S.T
        ev = torch.linalg.eigvalsh(G).flip(0)
        mean_dir = S.mean(0)
        share_mean = float(((S @ mean_dir) ** 2).sum() / (mean_dir @ mean_dir) / G.trace())
        cos = (S / S.norm(dim=1, keepdim=True)) @ (S / S.norm(dim=1, keepdim=True)).T
        off = cos[~torch.eye(len(sk), dtype=bool, device=cos.device)]
        rep[ck] = {"problems": len(sk), "eigen_share_top5": [float(x) for x in (ev[:5] / ev.sum())],
                   "mean_direction_share": share_mean, "mean_pairwise_cosine": float(off.mean()),
                   "pairwise_cosine_quartiles": [float(x) for x in torch.quantile(off, torch.tensor([0.25, 0.5, 0.75], device=off.device))],
                   "success_counts": [c for _, c in kept]}
        del runner
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
    if args.grad_ckpts:
        rep["gradients"] = grad_part(args, dev)
        print(json.dumps(rep["gradients"], indent=1), flush=True)
    (Path(args.out) / "direction_diag.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
