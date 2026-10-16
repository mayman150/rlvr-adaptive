"""Stage B: measure the four per-prompt numbers (p, |grad p|^2, F0, F1) in a real model.

For each prompt, sample N on-policy answers, verify them, and compute exact
per-answer score vectors in two declared parameter blocks (as in STOP):
  head  the output projection treated as its own matrix: token gradient
        (e_y - pi_t) h_t^T;
  mlp   the last decoder layer's MLP down-projection: token gradient
        delta_t a_t^T, exact from one backward pass per answer.
The answer score is the sum of its token gradients (no length normalization).
The answer Gram matrix Gamma_ij = <l_i, l_j> is formed exactly from token-level
factors. Estimates per prompt:
  F_r      mean of Gamma_ii over answers with reward r;
  G2       unbiased pair statistic mean_{i<j} (r_i - c)(r_j - c) Gamma_ij, valid for
           any c independent of the pair (reported for c = leave-pair-out mean
           and for c = 1/2);
  pair-type means of Gamma_ij for (correct, correct), (wrong, wrong), (mixed),
           which theory predicts equal G2/p^2, G2/(1-p)^2, -G2/(p(1-p)).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verifier import reward  # noqa: E402

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(tokenizer, question):
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM}\n\nQuestion: {question}\nAnswer:"


def trim_at_eos(seq, eos_id):
    hits = (seq == eos_id).nonzero()
    return seq[: int(hits[0]) + 1] if len(hits) else seq


@torch.no_grad()
def sample(model, prefix_ids, n, max_new, seed, eos_id):
    torch.manual_seed(seed)
    batch = prefix_ids.repeat(n, 1)
    out = model.generate(batch, attention_mask=torch.ones_like(batch), do_sample=True,
                         temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_new,
                         pad_token_id=eos_id)
    return [trim_at_eos(row[prefix_ids.shape[1]:], eos_id) for row in out]


def token_factors(model, prompt_ids, response):
    """Per-token gradient factors of one answer in the head and mlp blocks."""
    ids = torch.cat([prompt_ids[0], response])[None]
    n_prompt, T = prompt_ids.shape[1], len(response)
    pos = torch.arange(n_prompt - 1, n_prompt - 1 + T, device=ids.device)
    down = model.model.layers[-1].mlp.down_proj
    captured = {}

    def down_hook(module, inputs, output):
        captured["a"] = inputs[0].detach()[0]
        output.register_hook(lambda g: captured.__setitem__("d", g.detach()[0]))

    def head_hook(module, inputs, output):
        captured["h"] = inputs[0].detach()[0]

    handles = [down.register_forward_hook(down_hook), model.lm_head.register_forward_hook(head_hook)]
    try:
        model.zero_grad(set_to_none=True)
        logp = torch.log_softmax(model(input_ids=ids).logits[0].float()[pos], dim=-1)
        logp.gather(1, response[:, None]).sum().backward()
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
    with torch.no_grad():
        E = -logp.exp()
        rows = torch.arange(T, device=E.device)
        E[rows, response] = -torch.expm1(logp[rows, response])
    return {"E": E.detach(), "h": captured["h"][pos].float(), "a": captured["a"][pos].float(),
            "d": captured["d"][pos].float()}


def answer_gram(factors):
    n = len(factors)
    head, mlp = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            fi, fj = factors[i], factors[j]
            hv = float(((fi["E"] @ fj["E"].T) * (fi["h"] @ fj["h"].T)).cpu().double().sum())
            mv = float(((fi["d"] @ fj["d"].T) * (fi["a"] @ fj["a"].T)).cpu().double().sum())
            head[i, j] = head[j, i] = hv
            mlp[i, j] = mlp[j, i] = mv
    return head, mlp


def estimates(r, G):
    r = np.asarray(r, float)
    n, S = len(r), r.sum()
    p = S / n
    iu = np.triu_indices(n, 1)
    c_lpo = (S - r[iu[0]] - r[iu[1]]) / (n - 2)
    g2_lpo = float(np.mean((r[iu[0]] - c_lpo) * (r[iu[1]] - c_lpo) * G[iu]))
    g2_half = float(np.mean((r[iu[0]] - 0.5) * (r[iu[1]] - 0.5) * G[iu]))
    diag = np.diag(G)
    out = {"p": p, "G2_lpo": g2_lpo, "G2_half": g2_half,
           "F1": float(diag[r == 1].mean()) if (r == 1).any() else None,
           "F0": float(diag[r == 0].mean()) if (r == 0).any() else None}
    for name, mask in (("pair11", (r[iu[0]] == 1) & (r[iu[1]] == 1)),
                       ("pair00", (r[iu[0]] == 0) & (r[iu[1]] == 0)),
                       ("pair10", r[iu[0]] != r[iu[1]])):
        out[name] = float(G[iu][mask].mean()) if mask.any() else None
    cos = G[iu] / np.sqrt(diag[iu[0]] * diag[iu[1]])
    out["median_abs_cos"] = float(np.median(np.abs(cos)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--problems", default=str(HERE / "problems.json"))
    parser.add_argument("--answers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = json.loads(Path(args.problems).read_text())[: args.limit]
    offset = args.start
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    model.model.layers[-1].mlp.down_proj.weight.requires_grad_(True)
    eos = tokenizer.eos_token_id

    records = []
    for qi, prob in enumerate(problems):
        if qi < offset:
            continue
        start = time.time()
        prompt_ids = tokenizer(build_prompt(tokenizer, prob["question"]), return_tensors="pt").input_ids.to(args.device)
        responses = sample(model, prompt_ids, args.answers, args.max_tokens, args.seed + 7919 * qi, eos)
        rewards = [reward(tokenizer.decode(x), prob["answer"]) for x in responses]
        factors = [token_factors(model, prompt_ids, x) for x in responses]
        head, mlp = answer_gram(factors)
        del factors
        if args.device == "mps":
            torch.mps.empty_cache()
        rec = {"index": qi, "level": prob["level"], "subject": prob["subject"], "answer": prob["answer"],
               "rewards": rewards, "lengths": [len(x) for x in responses],
               "gram_head": head.tolist(), "gram_mlp": mlp.tolist(),
               "estimates": {name: estimates(rewards, G) for name, G in
                             (("head", head), ("mlp", mlp), ("combined", head + mlp))}}
        records.append(rec)
        e = rec["estimates"]["combined"]
        print(f"prompt {qi} ({prob['level']}): p={e['p']:.3f}, G2={e['G2_lpo']:.3g} (c=1/2: {e['G2_half']:.3g}), "
              f"F1={e['F1']}, F0={e['F0']}, median |cos|={e['median_abs_cos']:.3f}, {time.time() - start:.0f}s", flush=True)
        (out_dir / "summary.json").write_text(json.dumps(
            {"model": args.model, "answers": args.answers, "max_tokens": args.max_tokens,
             "seed": args.seed, "records": records}, indent=1))
    print(f"wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
