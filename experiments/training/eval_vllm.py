"""eval_ckpt.py with vLLM generation: the same options, outputs (<tag>_eval.json, <tag>_texts.json.gz) and checker,
about an order of magnitude faster for long answers. Each checkpoint is evaluated in its own subprocess (vLLM does not
release GPU memory reliably between engines).

Sampling: temperature T, top_p 1, no top-k, up to --max-new tokens, stop at the model's stop tokens, a fixed seed per
request; greedy: temperature 0. Answers are decoded without special tokens, as in eval_ckpt.py; an answer is truncated
if it ended because of the length limit.

Usage: python eval_vllm.py --ckpts DIR [--base MODEL] --problems FILE --n 32 --out DIR [eval_ckpt.py options]
"""

import argparse
import gzip
import json
import os
import math
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATOR = "eval_vllm v2"  # bump when the generation or accounting changes
FORCE_SUFFIX = "\n\nFinal answer: \\boxed{"
sys.path.insert(0, str(HERE))


def load_problems(path, limit, offset=0):
    p = Path(path) if Path(path).exists() else HERE / "data" / path
    return json.loads(p.read_text())[offset:][:limit]


def compat_tokenizer(path):
    """A copy of a checkpoint's tokenizer readable by transformers 4.x: checkpoints saved by transformers 5 store
    extra special tokens as a list ("extra_special_tokens") and add 5.x-only keys; 4.x expects
    additional_special_tokens. The vocabulary (tokenizer.json) and chat template are copied unchanged."""
    import hashlib
    import shutil
    src = Path(path)
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    if not isinstance(cfg.get("extra_special_tokens"), list) and "backend" not in cfg:
        return str(src)
    # one copy per process: lanes evaluating the same checkpoint would otherwise race on the files
    out = Path("/tmp") / ("tok_" + hashlib.sha1(str(src).encode()).hexdigest()[:12] + f"_{os.getpid()}")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "chat_template.jinja", "vocab.json", "merges.txt", "special_tokens_map.json"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    extra = cfg.pop("extra_special_tokens", None)
    if isinstance(extra, list):
        cfg["additional_special_tokens"] = extra
    for k in ("backend", "is_local", "local_files_only"):
        cfg.pop(k, None)
    (out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=1))
    return str(out)


def evaluate(job):
    """One checkpoint (run in a subprocess): sampled pass@k and, optionally, greedy accuracy."""
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"vllm {vllm.__version__}, transformers {transformers.__version__} ({transformers.__file__})", flush=True)

    from train_grpo import build_prompt, pass_at_k, stop_token_ids
    from verifier import last_boxed, reward

    t0 = time.time()
    a = job["args"]
    problems = load_problems(a["problems"], a["limit"], a["offset"])
    tok_dir = compat_tokenizer(job["path"])
    tok = AutoTokenizer.from_pretrained(tok_dir)
    assert tok.chat_template is not None, f"{job['path']}: no chat template (prompts would differ from eval_ckpt.py)"
    stop = stop_token_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else stop[0]
    texts = [build_prompt(tok, q["question"]) for q in problems] if job["sampled"] else []
    gp = load_problems(a["greedy_problems"], None) if job["greedy"] else []
    gtexts = [build_prompt(tok, q["question"]) for q in gp]
    longest = max([len(tok(t).input_ids) for t in texts + gtexts] or [0])
    from transformers import AutoConfig
    limit = getattr(AutoConfig.from_pretrained(job["path"]), "max_position_embeddings", 4096)
    force = a.get("force_answer", 0)
    suffix = tok(FORCE_SUFFIX, add_special_tokens=False).input_ids if force else []
    room = longest + a["max_new"] + (len(suffix) + force if force else 0)
    if room > limit:
        raise SystemExit(f"prompt {longest} + max_new {a['max_new']} (+ forced answer) exceeds the model's context {limit}")
    llm = LLM(model=job["path"], tokenizer=tok_dir, dtype="bfloat16", seed=a["seed"], gpu_memory_utilization=0.88,
              max_model_len=room, enable_prefix_caching=a["prefix_caching"], tensor_parallel_size=a.get("tp", 1),
              # vLLM's custom all-reduce needs CUDA IPC between GPUs, which SageMaker containers do not allow
              # ("custom_all_reduce.cuh: invalid argument"); NCCL all-reduce is used instead
              disable_custom_all_reduce=a.get("tp", 1) > 1,
              # CUDA graphs with NCCL collectives raised "illegal memory access" at tensor parallelism 8 on SageMaker
              enforce_eager=a.get("enforce_eager", False))

    def decode(o):
        """eval_ckpt.py's accounting on the generated token ids: cut after the first stop token, decode without
        special tokens; an answer is complete iff it contains a stop token."""
        ids = list(o.token_ids)
        cut = next((i + 1 for i, t in enumerate(ids) if t in stop), None)
        length = cut if cut is not None else len([t for t in ids if t != pad])
        ended = cut is not None or (o.finish_reason == "stop" and o.stop_reason in stop)  # stop token not echoed
        return tok.decode(ids[:cut or len(ids)], skip_special_tokens=True), length, ended

    res = {"tag": job["tag"], "model": job["path"], "n": a["n"], "problems": len(problems), "pass_at_k": {},
           "temperature": job["temperature"], "engine": f"vllm {vllm.__version__}", "evaluator": EVALUATOR,
           "fingerprint": job["fingerprint"], "tp": a.get("tp", 1)}
    if job["sampled"]:
        if job["temperature"] < 0.01:
            raise SystemExit("temperatures below 0.01 are clamped by vLLM; use the greedy path")
        # one seed per problem (siblings get seed + i inside vLLM), distinct across problems and shards
        sps = [SamplingParams(n=a["n"], temperature=job["temperature"], top_p=1.0, top_k=-1, max_tokens=a["max_new"],
                              stop_token_ids=stop, ignore_eos=True, seed=a["seed"] * 1000003 + (a["offset"] + i) * a["n"],
                              skip_special_tokens=True, detokenize=False)
               for i in range(len(texts))]
        gens = llm.generate(texts, sps)
        pids = [list(g.prompt_token_ids) for g in gens]  # the prompt ids vLLM actually used
        outs, meta, raw = [], [], []
        for g in gens:
            assert len(g.outputs) == a["n"]
            for o in g.outputs:
                text, length, ended = decode(o)
                outs.append(text)
                meta.append((length, ended))
                ids = list(o.token_ids)
                cut = next((i for i, t in enumerate(ids) if t in stop), None)
                raw.append(ids[:cut] if cut is not None else ids)
        res["mean_length_chars"] = sum(len(o) for o in outs) / len(outs)
        res["mean_length_tokens"] = sum(m[0] for m in meta) / len(meta)
        res["truncated_rate"] = sum(not m[1] for m in meta) / len(meta)
        res["boxed_rate"] = sum(last_boxed(o) is not None for o in outs) / len(outs)
        correct = [reward(o, problems[i // a["n"]]["answer"]) for i, o in enumerate(outs)]
        counts = [sum(correct[i * a["n"]:(i + 1) * a["n"]]) for i in range(len(problems))]
        ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= a["n"]]
        res.update({"counts": counts, "pass_at_k": {k: sum(pass_at_k(a["n"], c, k) for c in counts) / len(counts) for k in ks}})
        if force:
            # forced answer (Pre-registration 28): every sample without a final box (truncated, or ended without one)
            # is continued greedily for `force` tokens after FORCE_SUFFIX, then scored by the same verifier
            # control: the same samples continued greedily for the same number of tokens without the answer cue
            from vllm.inputs import TokensPrompt
            todo = [k for k, o in enumerate(outs) if last_boxed(o) is None]
            spf = SamplingParams(n=1, temperature=0.0, max_tokens=force, stop_token_ids=stop, ignore_eos=True,
                                 skip_special_tokens=True, detokenize=False)
            spc = SamplingParams(n=1, temperature=0.0, max_tokens=force + len(suffix), stop_token_ids=stop, ignore_eos=True,
                                 skip_special_tokens=True, detokenize=False)
            conts = llm.generate([TokensPrompt(prompt_token_ids=pids[k // a["n"]] + raw[k] + suffix) for k in todo], spf)
            ctrls = llm.generate([TokensPrompt(prompt_token_ids=pids[k // a["n"]] + raw[k]) for k in todo], spc)
            res["forced"] = {"tokens": force, "suffix": FORCE_SUFFIX, "suffix_tokens": len(suffix), "forced_samples": len(todo)}
            ftexts = {}
            for label, gens2, extra in (("forced", conts, suffix), ("extended", ctrls, [])):
                fouts = list(outs)
                for k, g in zip(todo, gens2):
                    fouts[k] = tok.decode(raw[k] + extra + list(g.outputs[0].token_ids), skip_special_tokens=True)
                    ftexts.setdefault(k, {})[label] = fouts[k][-600:]
                fcorrect = [reward(o, problems[i // a["n"]]["answer"]) for i, o in enumerate(fouts)]
                fcounts = [sum(fcorrect[i * a["n"]:(i + 1) * a["n"]]) for i in range(len(problems))]
                block = {"newly_right": sum(fcorrect[k] for k in todo), "counts": fcounts,
                         "pass_at_k": {k: sum(pass_at_k(a["n"], c, k) for c in fcounts) / len(fcounts) for k in ks}}
                if label == "forced":
                    res["forced"].update(block)
                else:
                    res["forced"]["equal_budget_control"] = block
            with gzip.open(Path(a["out"]) / f"{job['tag']}_forced_texts.json.gz", "wt") as fh:
                json.dump({"todo": todo, "tails": [ftexts[k] for k in todo]}, fh)
        with gzip.open(Path(a["out"]) / f"{job['tag']}_texts.json.gz", "wt") as fh:
            json.dump({"step": job["tag"], "n": a["n"], "problems": [{"question": q["question"], "answer": q["answer"]} for q in problems],
                       "samples": [{"problem": i // a["n"], "text": o, "reward": c} for i, (o, c) in enumerate(zip(outs, correct))]}, fh)
    if job["greedy"]:
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=a["max_new"], stop_token_ids=stop, ignore_eos=True,
                            skip_special_tokens=True, detokenize=False)
        go = [decode(g.outputs[0])[0] for g in llm.generate(gtexts, sp)]
        gc = [reward(o, q["answer"]) for o, q in zip(go, gp)]
        res["greedy"] = {"accuracy": sum(gc) / len(gc), "n": len(gc), "correct": gc}
    res["seconds"] = time.time() - t0
    (Path(a["out"]) / f"{job['tag']}_eval.json").write_text(json.dumps(res))
    print(f"{job['tag']}: pass@1 {100 * res['pass_at_k'].get(1, math.nan):.1f} pass@16 {100 * res['pass_at_k'].get(16, math.nan):.1f} "
          f"pass@32 {100 * res['pass_at_k'].get(32, math.nan):.1f}"
          + (f" greedy {100 * res['greedy']['accuracy']:.1f}" if "greedy" in res else "") + f" ({res['seconds']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--ckpts", default=None)
    ap.add_argument("--base", default=None, help="also evaluate this model as step 0")
    ap.add_argument("--problems", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--greedy-problems", default=None)
    ap.add_argument("--greedy-steps", default="")
    ap.add_argument("--sampled-steps", default="")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--no-prefix-caching", action="store_true", help="disable vLLM prefix caching (parity checks)")
    ap.add_argument("--entropy-prefixes", default=None, help="not supported here (use eval_ckpt.py)")
    ap.add_argument("--gen-batch", type=int, default=0, help="ignored (vLLM schedules its own batches)")
    ap.add_argument("--prefill-tokens", type=int, default=0, help="ignored")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda", help="ignored")
    ap.add_argument("--expect-steps", default="")
    ap.add_argument("--temperatures", default="")
    ap.add_argument("--base-temperatures", default="")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs (large models)")
    ap.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs (stability with tensor parallelism)")
    ap.add_argument("--identity", default="", help="model identity (e.g. repo@revision), part of the resume fingerprint")
    ap.add_argument("--force-answer", type=int, default=0, help="also score samples without a final box after a forced "
                    "answer of this many greedy tokens (0: off)")
    args = ap.parse_args()
    if args.worker:
        evaluate(json.loads(Path(args.worker).read_text()))
        return
    if not (args.ckpts and args.problems and args.out):
        raise SystemExit("--ckpts, --problems and --out are required")
    if args.entropy_prefixes:
        raise SystemExit("--entropy-prefixes is not supported by eval_vllm.py")
    args.prefix_caching = not args.no_prefix_caching
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    greedy_steps = {int(s) for s in args.greedy_steps.split(",") if s}
    sampled_steps = {int(s) for s in args.sampled_steps.split(",") if s}
    jobs = [(args.base, "step0", 0)] if args.base else []
    for d in sorted(Path(args.ckpts).glob("ckpt_step*"), key=lambda d: int(re.findall(r"\d+", d.name)[-1])):
        jobs.append((str(d), f"step{int(re.findall(r'[0-9]+', d.name)[-1])}", int(re.findall(r"\d+", d.name)[-1])))
    expected = {int(s) for s in args.expect_steps.split(",") if s}
    if expected - {s for _, _, s in jobs}:
        raise SystemExit(f"missing checkpoints for steps {sorted(expected - {s for _, _, s in jobs})} under {args.ckpts}")
    temps = [float(t) for t in args.temperatures.split(",") if t] or [None]
    base_temps = [float(t) for t in args.base_temperatures.split(",") if t] or temps
    shared = {k: getattr(args, k) for k in ("problems", "limit", "offset", "n", "greedy_problems", "max_new", "prefix_caching", "seed", "out",
                                            "force_answer", "tp", "enforce_eager")}
    for path, tag, step in jobs:
        sampled = not sampled_steps or step in sampled_steps
        if not sampled and step not in greedy_steps:
            continue
        for t in (base_temps if (args.base and step == 0 and path == args.base) else temps):
            full = tag if t is None else f"{tag}_T{t:.2f}"
            done = out_dir / f"{full}_eval.json"
            fingerprint = {k: shared[k] for k in ("problems", "limit", "offset", "n", "max_new", "seed")}
            if shared["force_answer"]:
                fingerprint["force_answer"] = shared["force_answer"]
            if args.identity:
                fingerprint["identity"] = args.identity
            if args.tp > 1:  # tensor parallelism changes numerics: never reuse results across settings
                fingerprint["tp"] = args.tp
            if done.exists():
                old = json.loads(done.read_text())
                if old.get("fingerprint") == fingerprint and old.get("evaluator") == EVALUATOR:
                    print(f"{full}: already done", flush=True)
                    continue
            job = {"path": str(path), "tag": full, "temperature": 1.0 if t is None else t, "sampled": sampled, "fingerprint": fingerprint,
                   "greedy": bool(args.greedy_problems) and step in greedy_steps and (t is None or abs(t - 1.0) < 1e-9), "args": shared}
            spec = out_dir / f".{full}_job.json"
            spec.write_text(json.dumps(job))
            code = subprocess.call([sys.executable, "-I", __file__, "--worker", str(spec)])
            if code != 0:
                raise SystemExit(f"{full}: worker exited with {code}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
