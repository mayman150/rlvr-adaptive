"""On-policy GRPO: mean-centered advantages versus failures scaled by w = F1 / F0.

Each step samples P prompts and G answers per prompt, verifies them, computes
advantages, and takes one gradient step (no importance ratios, no clipping of
ratios), with the loss
    -(1 / (P G L)) sum_i A_i sum_t log pi(y_it)
and a constant normalizer L = max_new_tokens (sequence sums, no length
normalization), as the theory assumes.

Arms:
  baseline  A_i = r_i - mean(r)                               (Dr. GRPO)
  scaled    A_i = (r_i - mean(r)) * (w if r_i = 0 else 1)
where w = F1 / F0 is the ratio of mean squared full-parameter gradient norms of
correct and wrong answers, measured once on the starting model before training.
Both arms generate exactly the same number of answers.
"""

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verifier import reward  # noqa: E402
import policy_table  # noqa: E402

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
STOP_TOKENS = ("<|endoftext|>", "<|im_end|>")


def build_prompt(tok, question):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class Runner:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tok = AutoTokenizer.from_pretrained(args.model)
        self.tok.padding_side = "left"
        self.stop_ids = [self.tok.convert_tokens_to_ids(t) for t in STOP_TOKENS]
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop_ids[0]
        self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(self.device)
        self.model.gradient_checkpointing_enable()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda"
                         else nullcontext())

    def trim(self, row):
        for k, t in enumerate(row.tolist()):
            if t in self.stop_ids:
                return row[: k + 1]
        return row

    @torch.no_grad()
    def generate(self, texts, n_each, max_new, seed, greedy=False):
        self.model.eval()
        items = [t for t in texts for _ in range(n_each)]
        outs = []
        torch.manual_seed(seed)
        for s in range(0, len(items), self.args.gen_batch):
            enc = self.tok(items[s:s + self.args.gen_batch], return_tensors="pt", padding=True).to(self.device)
            kwargs = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
            with self.autocast:
                gen = self.model.generate(**enc, max_new_tokens=max_new, eos_token_id=self.stop_ids,
                                          pad_token_id=self.pad_id, **kwargs)
            prompt_len = enc.input_ids.shape[1]
            for row, ids, mask in zip(gen, enc.input_ids, enc.attention_mask):
                outs.append((ids[mask.bool()], self.trim(row[prompt_len:])))
        return outs

    def logprob_sum(self, pairs):
        """Sum of response log-probabilities for each (prompt_ids, response_ids) pair."""
        seqs = [torch.cat([p, r]) for p, r in pairs]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, ((p, r), s) in enumerate(zip(pairs, seqs)):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
        with self.autocast:
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        sums = []
        for i, (p, r) in enumerate(pairs):
            rows = logits[i, len(p) - 1: len(p) - 1 + len(r)].float()
            sums.append((rows.gather(1, r[:, None]).squeeze(1) - torch.logsumexp(rows, dim=-1)).sum())
        return torch.stack(sums)

    def grad_sq_norm(self):
        return float(sum((p.grad.float() ** 2).sum() for p in self.params if p.grad is not None))

    def measure_ratio(self, problems, n_prompts, group, seed):
        """Mean squared full-parameter score norm of correct and wrong answers."""
        self.model.train()
        texts = [build_prompt(self.tok, q["question"]) for q in problems[:n_prompts]]
        answers = self.generate(texts, group, self.args.max_new, seed)
        norms = {0: [], 1: []}
        counts = [0] * n_prompts
        for k, (p, r) in enumerate(answers):
            q = problems[k // group]
            rew = reward(self.tok.decode(r), q["answer"])
            self.model.zero_grad(set_to_none=True)
            self.logprob_sum([(p, r)])[0].backward()
            norms[rew].append(self.grad_sq_norm())
            counts[k // group] += rew
        self.model.zero_grad(set_to_none=True)
        F1 = sum(norms[1]) / max(len(norms[1]), 1)
        F0 = sum(norms[0]) / max(len(norms[0]), 1)
        return {"F1": F1, "F0": F0, "n_correct": len(norms[1]), "n_wrong": len(norms[0]),
                "groups": [[c, group] for c in counts]}

    @torch.no_grad()
    def evaluate(self, eval_set, seed):
        texts = [build_prompt(self.tok, q["question"]) for q in eval_set]
        outs = self.generate(texts, 1, self.args.max_new, seed, greedy=True)
        correct = [reward(self.tok.decode(r), q["answer"]) for (_, r), q in zip(outs, eval_set)]
        return sum(correct) / len(correct)


def adaptive_collect(runner, args, arm, batch, decide, seed):
    """Sample `block` answers per active prompt per round until each prompt stops.

    decide(n0, n1) -> None to continue, or (a, b): advantages for every success
    and failure of that prompt (a = b = 0 abandons). Returns answers, advantages,
    rewards, per-prompt (successes, answers), rounds, and the number abandoned."""
    texts = [build_prompt(runner.tok, q["question"]) for q in batch]
    answers = [[] for _ in batch]
    rewards = [[] for _ in batch]
    final = [None] * len(batch)
    rnd = 0
    while any(f is None for f in final):
        active = [j for j, f in enumerate(final) if f is None]
        outs = runner.generate([texts[j] for j in active], args.block, args.max_new, seed + 7919 * rnd)
        for k, pair in enumerate(outs):
            j = active[k // args.block]
            answers[j].append(pair)
            rewards[j].append(reward(runner.tok.decode(pair[1]), batch[j]["answer"]))
        for j in active:
            n1 = sum(rewards[j])
            n0 = len(rewards[j]) - n1
            d = decide(n0, n1)
            if d is None and len(rewards[j]) >= args.cap:
                raise RuntimeError("rule did not stop at the cap")
            final[j] = d
        rnd += 1
    flat_answers, flat_adv, flat_rew = [], [], []
    for j in range(len(batch)):
        a, b = final[j]
        for pair, r in zip(answers[j], rewards[j]):
            flat_answers.append(pair)
            flat_rew.append(r)
            flat_adv.append(a if r == 1 else b)
    groups = [[sum(rs), len(rs)] for rs in rewards]
    abandoned = sum(1 for f in final if f == (0.0, 0.0))
    return flat_answers, flat_adv, flat_rew, groups, rnd, abandoned


def probe_rule(args):
    """SARA-style: abandon after the first block if all answers agree, else fill to the cap."""
    def decide(n0, n1):
        n = n0 + n1
        if n == args.block and (n0 == 0 or n1 == 0):
            return 0.0, 0.0
        if n >= args.cap:
            m = n1 / n
            return 1 - m, -m
        return None
    return decide


def table_rule(table, scale):
    def decide(n0, n1):
        d, a, b = table[(n0, n1)]
        return None if d == "continue" else (a * scale, b * scale)
    return decide


def table_scale(table):
    biggest = max(max(abs(a), abs(b)) for d, a, b in table.values() if d == "stop")
    return 1.0 / biggest if biggest > 0 else 1.0


def train_arm(args, arm, w, problems, eval_set, out_dir, calib=None):
    runner = Runner(args)
    opt = torch.optim.AdamW(runner.params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    rng = random.Random(args.seed)
    order = list(range(len(problems)))
    rng.shuffle(order)
    log, evals = [], []
    norm = args.budget * args.max_new
    calib_groups = (calib or {}).get("groups") or [[1, 8]]
    ratio = (calib["F0"] / calib["F1"]) if calib and calib.get("F1") else 1.0 / w
    spent, prompts_done, pointer, mean_n, recent = 0, 0, 0, float(args.group), []
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            acc = runner.evaluate(eval_set, args.seed)
            evals.append({"step": step, "accuracy": acc})
            print(f"[{arm}] step {step} eval accuracy {acc:.4f}", flush=True)
            (out_dir / f"{arm}_eval.json").write_text(json.dumps(evals, indent=1))
        if step == args.steps:
            break
        start = time.time()
        extra = {}
        if arm in ("baseline", "scaled"):
            batch = [problems[order[(step * args.prompts_per_step + j) % len(order)]] for j in range(args.prompts_per_step)]
            texts = [build_prompt(runner.tok, q["question"]) for q in batch]
            answers = runner.generate(texts, args.group, args.max_new, args.seed * 100003 + step)
            rewards = [reward(runner.tok.decode(r), batch[k // args.group]["answer"]) for k, (_, r) in enumerate(answers)]
            adv = []
            for j in range(args.prompts_per_step):
                g = rewards[j * args.group:(j + 1) * args.group]
                m = sum(g) / len(g)
                for r in g:
                    a = r - m
                    adv.append(a * w if (arm == "scaled" and r == 0) else a)
            groups = [[sum(rewards[j * args.group:(j + 1) * args.group]), args.group] for j in range(args.prompts_per_step)]
        else:
            if arm == "optimal" and step % args.resolve_every == 0:
                prior = policy_table.fit_beta(recent if recent else calib_groups)
                _, table = policy_table.solve_table(prior[0], prior[1], ratio, args.table_r, args.cap, args.block)
                scale = table_scale(table)
                decide = table_rule(table, scale)
                extra["prior"] = prior
            elif arm == "probe" and step == 0:
                decide = probe_rule(args)
            n_prompts = min(64, max(2, round((args.budget * (step + 1) - spent) / mean_n)))
            batch = [problems[order[(pointer + j) % len(order)]] for j in range(n_prompts)]
            pointer += n_prompts
            answers, adv, rewards, groups, rounds, abandoned = adaptive_collect(
                runner, args, arm, batch, decide, args.seed * 100003 + step)
            spent += len(answers)
            prompts_done += n_prompts
            mean_n = spent / prompts_done
            recent = (recent + groups)[-200:]
            extra.update({"prompts": n_prompts, "answers": len(answers), "rounds": rounds,
                          "cumulative_answers": spent, "abandoned": abandoned})
        runner.model.train()
        opt.zero_grad(set_to_none=True)
        live = [k for k, a in enumerate(adv) if a != 0]
        for s in range(0, len(live), args.train_batch):
            idx = live[s:s + args.train_batch]
            lp = runner.logprob_sum([answers[k] for k in idx])
            weights = torch.tensor([adv[k] for k in idx], device=runner.device) / norm
            (-(weights * lp).sum()).backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(runner.params, args.clip)) if live else 0.0
        if live:
            opt.step()
        mixed = sum(0 < k < n for k, n in groups)
        rec = {"step": step, "mean_reward": sum(rewards) / len(rewards), "mixed_groups": mixed,
               "groups": len(groups), "grad_norm": grad_norm, "answers": len(answers),
               "tokens": sum(len(r) for _, r in answers),
               "mean_length": sum(len(r) for _, r in answers) / len(answers),
               "seconds": time.time() - start, **extra}
        log.append(rec)
        print(f"[{arm}] step {step} reward {rec['mean_reward']:.3f} mixed {mixed}/{len(groups)} answers {len(answers)} "
              f"grad {grad_norm:.3g} len {rec['mean_length']:.0f} {rec['seconds']:.0f}s", flush=True)
        (out_dir / f"{arm}_train.json").write_text(json.dumps(log, indent=1))
    del runner, opt
    if args.device == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train", default=str(HERE / "data" / "train.json"))
    parser.add_argument("--eval", default=str(HERE / "data" / "eval_math500_numeric.json"))
    parser.add_argument("--arms", default="baseline,scaled")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--train-batch", type=int, default=4)
    parser.add_argument("--gen-batch", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--calib-prompts", type=int, default=16)
    parser.add_argument("--budget", type=int, default=None, help="answers per step; default prompts x group")
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--cap", type=int, default=16)
    parser.add_argument("--table-r", type=float, default=130.0)
    parser.add_argument("--resolve-every", type=int, default=10)
    parser.add_argument("--w", type=float, default=None, help="skip calibration and use this w")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.budget is None:
        args.budget = args.prompts_per_step * args.group
    problems = json.loads(Path(args.train).read_text())
    eval_set = json.loads(Path(args.eval).read_text())[: args.eval_limit]
    random.Random(args.seed).shuffle(problems)

    calib = None
    if args.w is None:
        runner = Runner(args)
        calib = runner.measure_ratio(problems[-args.calib_prompts:], args.calib_prompts, args.group, args.seed + 7)
        del runner
        w = min(1.0, max(0.1, calib["F1"] / calib["F0"])) if calib["F0"] > 0 and calib["F1"] > 0 else 1.0
        calib["w"] = w
        print(f"calibration: F1 {calib['F1']:.4g}, F0 {calib['F0']:.4g} over {calib['n_correct']} correct and "
              f"{calib['n_wrong']} wrong answers; w = {w:.3f}", flush=True)
        (out_dir / "calibration.json").write_text(json.dumps(calib, indent=1))
    else:
        w = args.w
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "w_used": w}, indent=1))
    for arm in args.arms.split(","):
        train_arm(args, arm, w, problems[: -args.calib_prompts], eval_set, out_dir, calib)
    print("done", flush=True)


if __name__ == "__main__":
    main()
