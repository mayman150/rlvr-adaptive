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
import math
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_problems(path, limit, offset=0):
    p = Path(path) if Path(path).exists() else HERE / "data" / path
    return json.loads(p.read_text())[offset:][:limit]


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
    tok = AutoTokenizer.from_pretrained(job["path"])
    stop = stop_token_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else stop[0]
    texts = [build_prompt(tok, q["question"]) for q in problems] if job["sampled"] else []
    gp = load_problems(a["greedy_problems"], None) if job["greedy"] else []
    gtexts = [build_prompt(tok, q["question"]) for q in gp]
    longest = max([len(tok(t).input_ids) for t in texts + gtexts] or [0])
    limit = json.loads((Path(job["path"]) / "config.json").read_text()).get("max_position_embeddings", 4096)
    if longest + a["max_new"] > limit:
        raise SystemExit(f"prompt {longest} + max_new {a['max_new']} exceeds the model's context {limit}")
    llm = LLM(model=job["path"], dtype="bfloat16", seed=a["seed"], gpu_memory_utilization=0.88,
              max_model_len=longest + a["max_new"], enable_prefix_caching=a["prefix_caching"])

    def decode(o):
        """eval_ckpt.py's accounting on the generated token ids: cut after the first stop token, decode without
        special tokens; an answer is complete iff it contains a stop token."""
        ids = list(o.token_ids)
        cut = next((i + 1 for i, t in enumerate(ids) if t in stop), None)
        length = cut if cut is not None else len([t for t in ids if t != pad])
        ended = cut is not None or (o.finish_reason == "stop" and o.stop_reason in stop)  # stop token not echoed
        return tok.decode(ids[:cut or len(ids)], skip_special_tokens=True), length, ended

    res = {"tag": job["tag"], "model": job["path"], "n": a["n"], "problems": len(problems), "pass_at_k": {},
           "temperature": job["temperature"], "engine": f"vllm {vllm.__version__}"}
    if job["sampled"]:
        if job["temperature"] < 0.01:
            raise SystemExit("temperatures below 0.01 are clamped by vLLM; use the greedy path")
        # one seed per problem (siblings get seed + i inside vLLM), distinct across problems and shards
        sps = [SamplingParams(n=a["n"], temperature=job["temperature"], top_p=1.0, top_k=-1, max_tokens=a["max_new"],
                              stop_token_ids=stop, ignore_eos=True, seed=a["seed"] * 1000003 + (a["offset"] + i) * a["n"],
                              skip_special_tokens=True, detokenize=False)
               for i in range(len(texts))]
        gens = llm.generate(texts, sps)
        outs, meta = [], []
        for g in gens:
            assert len(g.outputs) == a["n"]
            for o in g.outputs:
                text, length, ended = decode(o)
                outs.append(text)
                meta.append((length, ended))
        res["mean_length_chars"] = sum(len(o) for o in outs) / len(outs)
        res["mean_length_tokens"] = sum(m[0] for m in meta) / len(meta)
        res["truncated_rate"] = sum(not m[1] for m in meta) / len(meta)
        res["boxed_rate"] = sum(last_boxed(o) is not None for o in outs) / len(outs)
        correct = [reward(o, problems[i // a["n"]]["answer"]) for i, o in enumerate(outs)]
        counts = [sum(correct[i * a["n"]:(i + 1) * a["n"]]) for i in range(len(problems))]
        ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= a["n"]]
        res.update({"counts": counts, "pass_at_k": {k: sum(pass_at_k(a["n"], c, k) for c in counts) / len(counts) for k in ks}})
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
    shared = {k: getattr(args, k) for k in ("problems", "limit", "offset", "n", "greedy_problems", "max_new", "prefix_caching", "seed", "out")}
    for path, tag, step in jobs:
        sampled = not sampled_steps or step in sampled_steps
        if not sampled and step not in greedy_steps:
            continue
        for t in (base_temps if (args.base and step == 0 and path == args.base) else temps):
            full = tag if t is None else f"{tag}_T{t:.2f}"
            done = out_dir / f"{full}_eval.json"
            if done.exists() and str(json.loads(done.read_text()).get("engine", "")).startswith("vllm"):
                print(f"{full}: already done", flush=True)
                continue
            job = {"path": str(path), "tag": full, "temperature": 1.0 if t is None else t, "sampled": sampled,
                   "greedy": bool(args.greedy_problems) and step in greedy_steps and (t is None or abs(t - 1.0) < 1e-9), "args": shared}
            spec = out_dir / f".{full}_job.json"
            spec.write_text(json.dumps(job))
            code = subprocess.call([sys.executable, "-I", __file__, "--worker", str(spec)])
            if code != 0:
                raise SystemExit(f"{full}: worker exited with {code}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
