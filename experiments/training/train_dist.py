"""Multi-GPU on-policy GRPO for larger models (Qwen2.5-Math-7B): the plain on-policy loss of train_grpo.py
    -(1 / (P G L)) sum_i A_i sum_t log pi(y_it)
(one update per rollout batch, no importance ratios, sequence sums, constant normalizer P G L with
L = max_new) with FSDP2 training and data-parallel generation.

Each rank holds an FSDP2 shard of the fp32 training model and a full bf16 copy of the policy for generation,
refreshed after every update. By default the training forward runs under bf16 autocast on fp32 weights, as in
train_grpo.py (--param-dtype bf16 instead gathers bf16 weights: half the communication, bf16 residual stream). Rollouts are split across ranks one
answer at a time; rewards are gathered so that every rank computes the same group statistics, and the
answers with a non-zero advantage are redistributed so that every rank runs the same number of forward and
backward passes (FSDP collectives must match across ranks).

Launch: torchrun --nproc-per-node N train_dist.py --model DIR --out DIR [options]
Supported: the fixed-group arms of train_grpo.py (baseline = Dr. GRPO, psr, nsr, wreinforce, maxrl, ...),
and the two-stage allocation (--alloc-first, --alloc-value, --alloc-cap, --alloc-norm). Same data order
and output files (<arm>_train.json, config.json, ckpt_step<N>/) as train_grpo.py.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_grpo import (FIXED_ARMS, Runner, allocate_extra, allocation_value, build_prompt,  # noqa: E402
                        fixed_advantage, hora_allocation, law_value, stop_token_ids)
from verifier import last_boxed, reward  # noqa: E402


def view(args, tok, model, autocast=False):
    """A Runner over an existing model, for its generate and logprob_sum methods."""
    r = Runner.__new__(Runner)
    r.args, r.device, r.tok, r.model = args, "cuda", tok, model
    r.stop_ids = stop_token_ids(tok)
    r.pad_id = tok.pad_token_id if tok.pad_token_id is not None else r.stop_ids[0]
    r.autocast = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else nullcontext()
    return r


def load_models(args, dev):
    train = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, low_cpu_mem_usage=True)
    train.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train.config.use_cache = False
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if args.param_dtype == "bf16" else None,
                              reduce_dtype=torch.float32)
    # fully_shard moves each CPU-resident module to the GPU as it shards it (no full fp32 copy on one GPU).
    for layer in train.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    fully_shard(train, mesh=mesh, mp_policy=mp)
    assert all(p.device.type == "cuda" for p in train.parameters()) and all(b.device.type == "cuda" for b in train.buffers())
    torch.cuda.empty_cache()
    gen = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    return train, gen


@torch.no_grad()
def sync_weights(train, gen_params):
    """Copy the full training parameters into the bf16 generation model (one all-gather per parameter)."""
    for name, p in train.named_parameters():
        full = p.full_tensor() if isinstance(p, DTensor) else p
        gen_params[name].copy_(full.to(torch.bfloat16))


def gather(obj, world):
    out = [None] * world
    dist.all_gather_object(out, obj)
    return out


def rollouts(gen, args, batch, texts, slots, seed, rank, world):
    """Generate one answer for every (problem, slot) item; items are dealt to ranks round-robin.
    Returns {item: (reward, length, stopped, boxed)} for all items and {item: (prompt_ids, response_ids)}
    for this rank's own items."""
    mine = slots[rank::world]
    outs = gen.generate([texts[j] for j, _ in mine], 1, args.max_new, seed + 7919 * rank) if mine else []
    local, seqs = {}, {}
    for item, (p, r) in zip(mine, outs):
        text = gen.tok.decode(r)
        stopped = bool(len(r)) and int(r[-1]) in gen.stop_ids
        local[item] = (reward(text, batch[item[0]]["answer"]), len(r), stopped, last_boxed(text) is not None)
        seqs[item] = (p.cpu(), r.cpu())
    stats = {}
    for part in gather(local, world):
        stats.update(part)
    assert len(stats) == len(slots), "every item needs exactly one answer"
    return stats, seqs


def allocation(args, c1):
    P, n0 = len(c1), args.alloc_first
    left = args.prompts_per_step * args.group - P * n0
    if args.alloc_value == "hora":
        return hora_allocation(c1, n0, left, args.alloc_cap - n0)
    value = {"coverage": lambda c: allocation_value(c, n0, args.coverage_k),
             "law": lambda c: law_value(c, n0, args.coverage_k, args.alloc_lambda),
             "variance": lambda c: (c + 0.5) * (n0 - c + 0.5) / ((n0 + 1.0) * (n0 + 2.0))}[args.alloc_value]
    return allocate_extra([value(c) for c in c1], left, args.alloc_cap - n0)


def teacher_bank(args, tok, problems):
    """Teacher solutions (Pre-registration 29): question -> (prompt_ids, response_ids), the first of each problem's
    listed responses that survives Pre-registration 14's canonicalization (retokenized without special tokens, at most
    max_new - 1 tokens, verifies, has a final box; the stop token appended)."""
    path = Path(args.teacher_archive) if Path(args.teacher_archive).exists() else HERE / "data" / args.teacher_archive
    recs = {d["question"]: d for d in json.loads(path.read_text())}
    stops = stop_token_ids(tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    stop = im_end if isinstance(im_end, int) and im_end in stops else stops[0]
    bank = {}
    for q in problems:
        d = recs.get(q["question"])
        for resp in (d.get("responses") or ([d["response"]] if d.get("response") else [])) if d else []:
            ids = tok(resp, add_special_tokens=False).input_ids
            text = tok.decode(ids, skip_special_tokens=True)
            if len(ids) <= args.max_new - 1 and reward(text, q["answer"]) and last_boxed(text) is not None:
                bank[q["question"]] = (torch.tensor(tok(build_prompt(tok, q["question"])).input_ids), torch.tensor(ids + [stop]))
                break
    return bank


def deal(live, lengths, world):
    """Deal the live items to ranks: longest first, snake order (counts differ by at most one)."""
    order = sorted(live, key=lambda k: -lengths[k])
    out = [[] for _ in range(world)]
    for i, k in enumerate(order):
        lap, pos = divmod(i, world)
        out[pos if lap % 2 == 0 else world - 1 - pos].append(k)
    return out


def train_arm(args, arm, problems, out_dir, rank, world, dev):
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    train, gen_model = load_models(args, dev)
    gen, pol = view(args, tok, gen_model), view(args, tok, train, autocast=args.param_dtype == "fp32")
    gen_params = dict(gen_model.named_parameters())
    params = [p for p in train.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    rng = random.Random(args.seed)
    order = list(range(len(problems)))
    rng.shuffle(order)
    norm = args.prompts_per_step * args.group * args.max_new
    save_steps = {int(s) for s in args.save_steps.split(",") if s}
    pps, G, log = args.prompts_per_step, args.group, []
    bank = teacher_bank(args, tok, problems) if args.teacher_archive else None
    if bank is not None and rank == 0:
        print(f"[{arm}] teacher solutions for {len(bank)} of {len(problems)} problems (lambda {args.teacher_lambda}, "
              f"groups with at most {args.teacher_max_success} successes)", flush=True)
    dummy = (torch.tensor([pol.pad_id], device=dev), torch.tensor([pol.pad_id], device=dev))
    for step in range(args.steps + 1):
        if step in save_steps:
            if rank == 0:
                gen_model.save_pretrained(out_dir / f"ckpt_step{step}", safe_serialization=True)
                tok.save_pretrained(out_dir / f"ckpt_step{step}")
                print(f"[{arm}] saved checkpoint at step {step}", flush=True)
            dist.barrier()
        if step == args.steps:
            break
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        batch = [problems[order[(step * pps + j) % len(order)]] for j in range(pps)]
        texts = [build_prompt(tok, q["question"]) for q in batch]
        seed = args.seed * 100003 + step
        extra = {}
        if args.alloc_first > 0:
            n0 = args.alloc_first
            stats, seqs = rollouts(gen, args, batch, texts, [(j, s) for j in range(pps) for s in range(n0)], seed, rank, world)
            c1 = [sum(stats[(j, s)][0] for s in range(n0)) for j in range(pps)]
            more = allocation(args, c1)
            assert sum(more) == pps * G - pps * n0, "allocation must use the whole budget"
            stats2, seqs2 = rollouts(gen, args, batch, texts, [(j, n0 + t) for j in range(pps) for t in range(more[j])],
                                     seed + 104729, rank, world)
            stats.update(stats2)
            seqs.update(seqs2)
            sizes = [n0 + e for e in more]
            extra["alloc"] = {"sizes": sizes, "first_counts": c1, "max": max(sizes), "min": min(sizes)}
        else:
            stats, seqs = rollouts(gen, args, batch, texts, [(j, s) for j in range(pps) for s in range(G)], seed, rank, world)
            sizes = [G] * pps
        gen_seconds = time.time() - start
        assert sum(sizes) == pps * G == len(stats)
        adv, groups = {}, []
        for j, n_j in enumerate(sizes):
            g = [stats[(j, s)][0] for s in range(n_j)]
            m = sum(g) / n_j
            scale = G / n_j if (args.alloc_first > 0 and args.alloc_norm == "group") else 1.0
            for s, r in enumerate(g):
                adv[(j, s)] = scale * fixed_advantage(arm, r, m, args.w, args.pos_weight, args.shift, n_j, args.train_k)
            groups.append([sum(g), n_j])
        t_items = {}
        if bank is not None:
            # teacher term: -lambda log pi(y_T | x) / norm for the step's problems with a teacher solution whose group
            # has at most teacher_max_success successes; item (j, -1), dealt and microbatched like an answer
            for j, q in enumerate(batch):
                if q["question"] in bank and groups[j][0] <= args.teacher_max_success:
                    t_items[(j, -1)] = bank[q["question"]]
                    adv[(j, -1)] = args.teacher_lambda
            extra["teacher"] = {"solutions": len(t_items), "tokens": sum(len(v[1]) for v in t_items.values())}
        live = sorted(k for k, a in adv.items() if a != 0)
        lengths = {k: stats[k][1] for k in stats}
        lengths.update({k: len(v[1]) for k, v in t_items.items()})
        plan = deal(live, lengths, world)
        # Move the token ids of the live answers to the ranks that will run their backward passes.
        send = [{k: seqs[k] for k in plan[d] if k in seqs} for d in range(world)]
        own = {}
        for part in gather(send, world):
            own.update(part[rank])
        own.update({k: t_items[k] for k in plan[rank] if k in t_items})  # every rank holds the teacher bank
        assert set(own) == set(plan[rank]), "every live answer must reach its rank"
        if args.offload_gen:  # 48 GB GPUs: the generation copy waits in host memory during the update
            gen_model.to("cpu")
            torch.cuda.empty_cache()
        train.train()
        opt.zero_grad(set_to_none=True)
        n_micro = max(math.ceil(len(p) / args.train_batch) for p in plan) if live else 0
        mine = plan[rank]
        for b in range(n_micro):
            chunk = mine[b * args.train_batch:(b + 1) * args.train_batch]
            pairs = [(own[k][0].to(dev), own[k][1].to(dev)) for k in chunk] or [dummy]
            weights = torch.tensor([adv[k] for k in chunk] or [0.0], device=dev)
            lp = pol.logprob_sum(pairs)
            # FSDP averages gradients over ranks; the factor `world` turns the average into the global sum.
            (-(weights * lp).sum() * world / norm).backward()
        grad_norm = 0.0
        if live:
            total = torch.nn.utils.clip_grad_norm_(params, args.clip)
            grad_norm = float(total.full_tensor() if isinstance(total, DTensor) else total)
            if math.isfinite(grad_norm):
                opt.step()
            else:
                print(f"[{arm}] step {step}: non-finite gradient norm, update skipped", flush=True)
                extra["nonfinite_grad"] = True
        opt.zero_grad(set_to_none=True)
        if args.offload_gen:
            gen_model.to(dev)
        if live and "nonfinite_grad" not in extra:
            sync_weights(train, gen_params)
        rewards = [stats[(j, s)][0] for j, n_j in enumerate(sizes) for s in range(n_j)]
        first = [stats[(j, s)][0] for j, n_j in enumerate(sizes) for s in range(min(n_j, args.ref_block))]
        peak = torch.tensor(torch.cuda.max_memory_allocated() / 1e9, device=dev)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        rec = {"step": step, "mean_reward": sum(rewards) / len(rewards),
               "boxed_rate": sum(v[3] for v in stats.values()) / len(stats),
               "truncated_rate": sum(1 for v in stats.values() if v[1] >= args.max_new and not v[2]) / len(stats),
               "first_block_reward": sum(first) / len(first), "mixed_groups": sum(0 < c < n for c, n in groups),
               "groups": len(groups), "grad_norm": grad_norm, "answers": len(stats),
               "tokens": sum(lengths.values()), "mean_length": sum(lengths.values()) / len(stats),
               "group_counts": groups, "live": len(live), "micro_batches": n_micro, "peak_gb": float(peak),
               "gen_seconds": gen_seconds, "seconds": time.time() - start, "world": world, **extra}
        log.append(rec)
        if rank == 0:
            print(f"[{arm}] step {step} reward {rec['mean_reward']:.3f} mixed {rec['mixed_groups']}/{len(groups)} "
                  f"grad {grad_norm:.3g} len {rec['mean_length']:.0f} gen {gen_seconds:.0f}s total {rec['seconds']:.0f}s "
                  f"peak {rec['peak_gb']:.1f} GB", flush=True)
            (out_dir / f"{arm}_train.json").write_text(json.dumps(log, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default="train.json")
    ap.add_argument("--arms", default="baseline")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--train-batch", type=int, default=2)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--prefill-tokens", type=int, default=256)
    ap.add_argument("--save-steps", default="")
    ap.add_argument("--w", type=float, default=None, help="failure weight of the scaled arm (required for it)")
    ap.add_argument("--offload-gen", action="store_true",
                    help="keep the bf16 generation copy in host memory during each update (for 48 GB GPUs)")
    ap.add_argument("--param-dtype", default="fp32", choices=("fp32", "bf16"),
                    help="fp32: bf16 autocast on fp32 weights (train_grpo.py); bf16: FSDP gathers bf16 weights")
    ap.add_argument("--pos-weight", type=float, default=0.1)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--train-k", type=int, default=4)
    ap.add_argument("--ref-block", type=int, default=4)
    ap.add_argument("--calib-prompts", type=int, default=16, help="held back from training, as in train_grpo.py")
    ap.add_argument("--alloc-first", type=int, default=0)
    ap.add_argument("--alloc-cap", type=int, default=32)
    ap.add_argument("--alloc-value", default="coverage", choices=("coverage", "law", "variance", "hora"))
    ap.add_argument("--alloc-lambda", type=float, default=0.37)
    ap.add_argument("--alloc-norm", default="global", choices=("global", "group"))
    ap.add_argument("--coverage-k", type=int, default=16)
    ap.add_argument("--teacher-archive", default="", help="teacher solutions (discovery format) for teacher distillation")
    ap.add_argument("--teacher-lambda", type=float, default=1.0)
    ap.add_argument("--teacher-max-success", type=int, default=8, help="use a teacher solution only when the problem's "
                    "group has at most this many successes")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()  # unsupported train_grpo.py options are errors, not silently ignored
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert args.alloc_first == 0 or args.alloc_first <= args.group <= args.alloc_cap, \
        "allocation needs alloc_first <= group <= alloc_cap"
    arms = args.arms.split(",")
    assert "scaled" not in arms or args.w is not None, "the scaled arm needs an explicit --w"
    assert not (args.alloc_first > 0 and "passk" in arms), "passk advantages need fixed groups larger than train_k"
    assert not args.teacher_archive or (arms == ["baseline"] and args.alloc_first == 0 and args.teacher_lambda > 0), \
        "teacher distillation is implemented for plain Dr. GRPO"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.train) if Path(args.train).exists() else HERE / "data" / args.train
    problems = json.loads(path.read_text())
    random.Random(args.seed).shuffle(problems)
    if rank == 0:
        (out_dir / "config.json").write_text(json.dumps({**vars(args), "world": world, "w_used": args.w}, indent=1))
    problems = problems[: len(problems) - args.calib_prompts]  # the reference holds the last ones back
    assert problems, "no training problems"
    for arm in arms:
        assert arm in FIXED_ARMS and arm != "psrnoise", f"{arm} is not supported by train_dist.py"
        train_arm(args, arm, problems, out_dir, rank, world, dev)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("done", flush=True)


if __name__ == "__main__":
    main()
