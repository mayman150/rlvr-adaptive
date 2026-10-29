"""Pre-registration 30, Stage 1 feature: mean token NLL of each start on the validation problems' human solutions (one
frozen corpus, data/val_frontier_256_human.json, MATH test solutions from EleutherAI/hendrycks_math): our prompt, then
the human solution text retokenized without special tokens and truncated to 1,023 tokens, then the stop token (the
same one sft_start.py appends); response tokens only; token-averaged over the corpus.
Usage: python p30_nll.py --model BASE --starts NAME=DIR ... --out DIR   (the base itself is scored as "base")
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_grpo import Runner, build_prompt  # noqa: E402


def score(model, corpus, batch=8):
    runner = Runner(argparse.Namespace(model=model, device="cuda", gen_batch=8, prefill_tokens=256, max_new=1024, train_batch=batch))
    tok = runner.tok
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    stop = im_end if isinstance(im_end, int) and im_end in runner.stop_ids else runner.stop_ids[0]
    pairs = [(torch.tensor(tok(build_prompt(tok, d["question"])).input_ids),
              torch.tensor(tok(d["solution"], add_special_tokens=False).input_ids[:1023] + [stop])) for d in corpus]
    tot, n = 0.0, 0
    runner.model.eval()
    with torch.no_grad():
        for s in range(0, len(pairs), batch):
            for lp in runner.token_logprobs([(p.to("cuda"), r.to("cuda")) for p, r in pairs[s:s + batch]]):
                tot -= float(lp.sum())
                n += lp.numel()
    del runner
    torch.cuda.empty_cache()
    return tot / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--starts", nargs="+", default=[])
    ap.add_argument("--corpus", default=str(HERE / "data" / "val_frontier_256_human.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    corpus = json.loads(Path(a.corpus).read_text())
    assert len(corpus) == 255 and all(d["solution"] for d in corpus)
    res = {}
    for name, path in [("base", a.model)] + [s.split("=", 1) for s in a.starts]:
        nll, n = score(path, corpus)
        res[name] = {"mean_token_nll": nll, "tokens": n, "path": path}
        print(f"{name}: mean token NLL {nll:.4f} over {n} tokens", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "human_nll.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
