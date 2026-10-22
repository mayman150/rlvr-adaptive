"""Check 17: per-token log-likelihood of held-out correct and wrong base-model answers under checkpoints.

Usage: python transfer_probe.py --ckpts DIR [--base MODEL] --traces transfer_traces.json --steps 0,50,100,150 --out DIR
Writes OUT/probe_step<N>.json: {"step", "mean_logprob": [one value per trace, in the traces file's order]}.
Step 0 is the base model (--base) unless DIR has ckpt_step0 (the warm start's post-fine-tuning checkpoint).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_grpo import build_prompt  # noqa: E402


@torch.no_grad()
def score(model, tok, traces, device, batch=8):
    out = []
    for s in range(0, len(traces), batch):
        chunk = traces[s:s + batch]
        pairs = [(tok(build_prompt(tok, t["question"])).input_ids, tok(t["text"], add_special_tokens=False).input_ids) for t in chunk]
        width = max(len(p) + len(r) for p, r in pairs)
        ids = torch.full((len(pairs), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, (p, r) in enumerate(pairs):
            ids[i, :len(p) + len(r)] = torch.tensor(p + r)
            mask[i, :len(p) + len(r)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        for i, (p, r) in enumerate(pairs):
            rows = logits[i, len(p) - 1:len(p) - 1 + len(r)].float()
            lp = rows.gather(1, torch.tensor(r, device=device)[:, None]).squeeze(1) - torch.logsumexp(rows, dim=-1)
            out.append(float(lp.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--traces", default="transfer_traces.json")
    ap.add_argument("--steps", default="0,50,100,150")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    path = Path(args.traces) if Path(args.traces).exists() else HERE / "data" / args.traces
    traces = json.loads(path.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for step in [int(s) for s in args.steps.split(",")]:
        ck = Path(args.ckpts) / f"ckpt_step{step}"
        src = str(ck) if ck.exists() else (args.base if step == 0 else None)
        if src is None:
            raise SystemExit(f"no checkpoint for step {step} in {args.ckpts}")
        tok = AutoTokenizer.from_pretrained(src)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).to(args.device).eval()
        vals = score(model, tok, traces, args.device)
        (out / f"probe_step{step}.json").write_text(json.dumps({"step": step, "source": src, "mean_logprob": vals}))
        print(f"step {step} ({src}): {len(vals)} traces scored", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
