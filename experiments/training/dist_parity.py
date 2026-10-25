"""Launch check for train_dist.py (Pre-registration 20): on identical frozen rollouts, does the FSDP2 update
equal train_grpo.py's single-GPU update?

Rollouts: data/parity_rollouts.json (8 problems x 8 saved answers, fixed rewards), Dr. GRPO advantages within
each group, constant normalizer 64 x max_new. Reference, on rank 0 before FSDP: train_grpo.py's model (fp32
weights, bf16 autocast, gradient checkpointing) and its backward_weighted loss, then clipping and one AdamW
step. FSDP: train_dist.py's models and loss (the world factor, live answers dealt by deal(), dummy passes on
short ranks). Reported: global gradient-norm relative difference; for selected parameters the cosine and
relative L2 error of the raw gradient and of the AdamW update; and the mean absolute per-token log-probability
difference between the bf16 generation copy and the training model.

Launch: torchrun --nproc-per-node N dist_parity.py --model DIR [--out FILE]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_dist import deal, load_models, view  # noqa: E402
from train_grpo import backward_weighted, build_prompt  # noqa: E402

PICK = ("model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight")


def full(t):
    return (t.full_tensor() if isinstance(t, DTensor) else t).detach().float().cpu()


def compare(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    cos = float(a @ b / (a.norm() * b.norm())) if a.norm() > 0 and b.norm() > 0 else float("nan")
    return {"cosine": cos, "rel_l2": float((a - b).norm() / a.norm()) if a.norm() > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--train-batch", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--prefill-tokens", type=int, default=256)
    ap.add_argument("--param-dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref-repeat", action="store_true", help="also report the reference's own batching noise floor")
    args = ap.parse_args()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    rank, world = dist.get_rank(), dist.get_world_size()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    groups = json.loads((HERE / "data" / "parity_rollouts.json").read_text())["groups"]
    answers, adv = [], []
    for g in groups:
        p = torch.tensor(tok(build_prompt(tok, g["question"])).input_ids)
        m = sum(g["rewards"]) / len(g["rewards"])
        for text, r in zip(g["responses"], g["rewards"]):
            answers.append((p, torch.tensor(tok(text, add_special_tokens=False).input_ids[: args.max_new])))
            adv.append(r - m)
    norm = len(answers) * args.max_new
    live = [k for k, a in enumerate(adv) if a != 0]
    report = {"world": world, "answers": len(answers), "live": len(live), "param_dtype": args.param_dtype}

    def reference(order):
        """train_grpo.py's single-GPU update with the live answers in the given order (fresh model)."""
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
        model.gradient_checkpointing_enable()
        model.train()
        runner = view(args, tok, model, autocast=True)
        backward_weighted(runner, args, [(a.to(dev), b.to(dev)) for a, b in answers], order, adv, norm)
        named = dict(model.named_parameters())
        out = {"grad": {n: named[n].grad.detach().float().cpu() for n in PICK if n in named}}
        params = [q for q in model.parameters() if q.requires_grad]
        out["norm"] = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        before = {n: named[n].detach().float().cpu() for n in out["grad"]}
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
        opt.step()
        out["update"] = {n: named[n].detach().float().cpu() - before[n] for n in before}
        del model, opt, params, named, runner
        torch.cuda.empty_cache()
        return out

    ref = {}
    if rank == 0:
        ref = reference(live)
        if args.ref_repeat:  # noise floor: the same update with the answers batched in reverse order
            ref2 = reference(list(reversed(live)))
            report["reference_floor"] = {"grad_norm_rel_diff": abs(ref2["norm"] - ref["norm"]) / ref["norm"],
                                         "grad": {n: compare(ref["grad"][n], ref2["grad"][n]) for n in ref["grad"]},
                                         "update": {n: compare(ref["update"][n], ref2["update"][n]) for n in ref["update"]}}
            del ref2
    dist.barrier()

    train, gen_model = load_models(args, dev)  # train_dist.py's update
    pol, gen = view(args, tok, train, autocast=args.param_dtype == "fp32"), view(args, tok, gen_model)
    train.train()
    plan = deal(live, {k: len(answers[k][1]) for k in range(len(answers))}, world)
    mine = plan[rank]
    n_micro = max(math.ceil(len(p) / args.train_batch) for p in plan)
    dummy = (torch.tensor([pol.pad_id], device=dev), torch.tensor([pol.pad_id], device=dev))
    for b in range(n_micro):
        chunk = mine[b * args.train_batch:(b + 1) * args.train_batch]
        pairs = [(answers[k][0].to(dev), answers[k][1].to(dev)) for k in chunk] or [dummy]
        weights = torch.tensor([adv[k] for k in chunk] or [0.0], device=dev)
        (-(weights * pol.logprob_sum(pairs)).sum() * world / norm).backward()
    named = dict(train.named_parameters())
    grads = {n: full(named[n].grad) for n in PICK if n in named}
    params = [q for q in train.parameters() if q.requires_grad]
    total = torch.nn.utils.clip_grad_norm_(params, args.clip)
    fsdp_norm = float(total.full_tensor() if isinstance(total, DTensor) else total)
    before = {n: full(named[n]) for n in grads}
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    opt.step()
    update = {n: full(named[n]) - before[n] for n in grads}
    with torch.no_grad():  # generation copy (bf16 weights) against the training model, first 8 answers
        sample = [(a.to(dev), b.to(dev)) for a, b in answers[:8]]
        lp_train = pol.logprob_tokens(sample)
        lp_gen = gen.logprob_tokens(sample)
        diff = torch.cat([(x - y).abs() for x, y in zip(lp_train, lp_gen)])
    if rank == 0:
        report["grad_norm"] = {"reference": ref["norm"], "fsdp": fsdp_norm,
                               "rel_diff": abs(fsdp_norm - ref["norm"]) / ref["norm"]}
        report["grad"] = {n: compare(ref["grad"][n], grads[n]) for n in grads}
        report["update"] = {n: compare(ref["update"][n], update[n]) for n in update}
        report["gen_vs_train_logprob"] = {"mean_abs_per_token": float(diff.mean()), "max_abs": float(diff.max()),
                                          "tokens": int(diff.numel())}
        print(json.dumps(report, indent=1), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
