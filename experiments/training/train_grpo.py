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
import math
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
    if tok.chat_template is None:
        return f"{SYSTEM}\n\n{question}\n\n"
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
        budget = self.args.gen_batch * self.args.prefill_tokens
        s = 0
        while s < len(items):
            b = self.args.gen_batch
            enc = self.tok(items[s:s + b], return_tensors="pt", padding=True)
            while b > 1 and enc.input_ids.shape[1] * b > budget:
                b = max(1, budget // enc.input_ids.shape[1])
                enc = self.tok(items[s:s + b], return_tensors="pt", padding=True)
            enc = enc.to(self.device)
            s += b
            kwargs = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
            with self.autocast:
                gen = self.model.generate(**enc, max_new_tokens=max_new, eos_token_id=self.stop_ids,
                                          pad_token_id=self.pad_id, **kwargs)
            prompt_len = enc.input_ids.shape[1]
            for row, ids, mask in zip(gen, enc.input_ids, enc.attention_mask):
                outs.append((ids[mask.bool()], self.trim(row[prompt_len:])))
        return outs

    def logprob_sum(self, pairs, entropy=None):
        """Sum of response log-probabilities for each (prompt_ids, response_ids) pair.

        If `entropy` is a list, the summed token entropies of each response are appended to it."""
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
            lse = torch.logsumexp(rows, dim=-1)
            sums.append((rows.gather(1, r[:, None]).squeeze(1) - lse).sum())
            if entropy is not None:
                with torch.no_grad():
                    z = rows.detach()
                    entropy.append(float((lse.detach() - (torch.softmax(z, dim=-1) * z).sum(-1)).sum()))
        return torch.stack(sums)

    def logprob_tokens(self, pairs):
        """Per-token response log-probabilities for each (prompt_ids, response_ids) pair."""
        seqs = [torch.cat([p, r]) for p, r in pairs]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, ((p, r), s) in enumerate(zip(pairs, seqs)):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
        with self.autocast:
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        out = []
        for i, (p, r) in enumerate(pairs):
            rows = logits[i, len(p) - 1: len(p) - 1 + len(r)].float()
            out.append(rows.gather(1, r[:, None]).squeeze(1) - torch.logsumexp(rows, dim=-1))
        return out

    def grad_snapshot(self):
        return [p.grad.detach().to("cpu", copy=True) if p.grad is not None else torch.zeros(p.shape)
                for p in self.params]

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
    def evaluate(self, eval_set, seed, k=1, greedy=True):
        """Accuracy over k answers per problem, and the number correct for each problem."""
        texts = [build_prompt(self.tok, q["question"]) for q in eval_set]
        outs = self.generate(texts, k, self.args.max_new, seed, greedy=greedy)
        correct = [reward(self.tok.decode(r), eval_set[i // k]["answer"]) for i, (_, r) in enumerate(outs)]
        per_problem = [sum(correct[i * k:(i + 1) * k]) for i in range(len(eval_set))]
        return sum(correct) / len(correct), per_problem


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


FIXED_ARMS = ("baseline", "scaled", "psr", "nsr", "wreinforce")


def fixed_advantage(arm, r, m, w, pos_weight):
    """Advantage of one answer with reward r in a fixed group with mean reward m.

    psr rewards only correct answers, nsr penalizes only wrong ones, and wreinforce
    weights correct answers by pos_weight (Zhu et al. 2025). On-policy, psr and nsr have
    exactly the same expected update, n grad p per prompt."""
    if arm == "psr":
        return float(r)
    if arm == "nsr":
        return float(r) - 1.0
    if arm == "wreinforce":
        return pos_weight if r else -1.0
    a = r - m
    return a * w if (arm == "scaled" and r == 0) else a


def pass_at_k(n, c, k):
    """Unbiased pass@k from c correct out of n samples."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


@torch.no_grad()
def probe_entropy(runner, args, answers, rewards, owner):
    """Token entropy and mean token log-probability of the first ref_block answers of
    every prompt (an on-policy sample whatever the arm), split by correctness."""
    seen, idx = {}, []
    for k, j in enumerate(owner):
        seen[j] = seen.get(j, 0) + 1
        if seen[j] <= args.ref_block:
            idx.append(k)
    out = {1: [0.0, 0.0, 0], 0: [0.0, 0.0, 0]}
    for s in range(0, len(idx), args.train_batch):
        chunk = idx[s:s + args.train_batch]
        ent = []
        lps = runner.logprob_sum([answers[k] for k in chunk], ent)
        for k, e, lp in zip(chunk, ent, lps.tolist()):
            o = out[int(rewards[k])]
            o[0] += e
            o[1] += lp
            o[2] += len(answers[k][1])
    tot_tokens = out[1][2] + out[0][2]
    res = {"entropy": (out[1][0] + out[0][0]) / tot_tokens if tot_tokens else None}
    for r, name in ((1, "correct"), (0, "wrong")):
        if out[r][2]:
            res[f"entropy_{name}"] = out[r][0] / out[r][2]
            res[f"logprob_{name}"] = out[r][1] / out[r][2]
    return res


def backward_weighted(runner, args, answers, idx, weights, norm, entropy=None):
    """Accumulate the gradient of -sum_k weights[k] log pi(answer_k) / norm into .grad."""
    for s in range(0, len(idx), args.train_batch):
        chunk = idx[s:s + args.train_batch]
        lp = runner.logprob_sum([answers[k] for k in chunk], entropy)
        wt = torch.tensor([weights[k] for k in chunk], device=runner.device) / norm
        (-(wt * lp).sum()).backward()


def backward_surrogate(runner, args, answers, idx, adv, old_lp, norm):
    """PPO-style clipped surrogate over tokens; with old_lp from the current policy (ratio 1)
    and no length normalization this equals the sequence-sum REINFORCE loss."""
    for s in range(0, len(idx), args.train_batch):
        chunk = idx[s:s + args.train_batch]
        new = runner.logprob_tokens([answers[k] for k in chunk])
        total = 0.0
        for k, lp in zip(chunk, new):
            ratio = torch.exp(lp - old_lp[k])
            a = adv[k]
            surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * a)
            term = surr.mean() * args.max_new if args.length_norm else surr.sum()
            total = total + term
        (-total / norm).backward()


@torch.no_grad()
def old_logprobs(runner, args, answers):
    out = []
    for s in range(0, len(answers), args.train_batch):
        out.extend(t.detach() for t in runner.logprob_tokens(answers[s:s + args.train_batch]))
    return out


def diagnostic_step(runner, args, answers, adv, rewards, owner, norm, step, entropy):
    """Backward pass with split-half gradient statistics.

    Prompts are split at random into halves A and B. g_A and g_B are the arm's gradient
    from each half; ref_B is the Dr. GRPO estimator on the first `ref_block` answers of
    each prompt in B, an unbiased direction for the population gradient that is
    independent of g_A. Returns statistics; .grad holds g_A + g_B, as without diagnostics."""
    prompts = sorted(set(owner))
    perm = prompts[:]
    random.Random(args.seed * 1000003 + step).shuffle(perm)
    half_a = set(perm[: len(perm) // 2])
    first = {}
    for k, j in enumerate(owner):
        first.setdefault(j, [])
        if len(first[j]) < args.ref_block:
            first[j].append(k)
    ref_w = {}
    for j in prompts:
        if j in half_a:
            continue
        m = sum(rewards[k] for k in first[j]) / len(first[j])
        for k in first[j]:
            if rewards[k] != m:
                ref_w[k] = rewards[k] - m
    runner.model.zero_grad(set_to_none=True)
    backward_weighted(runner, args, answers, sorted(ref_w), ref_w, norm)
    ref = runner.grad_snapshot()
    ref2 = None
    if args.within_ref:
        # Second reference from answers ref_block+1 .. 2 ref_block of the same prompts, with its
        # own mean: independent of ref given the prompts, so <ref, ref2> measures the
        # per-prompt signal |grad p_x|^2 and not only the shared part.
        second = {}
        for k, j in enumerate(owner):
            if j not in half_a:
                second.setdefault(j, [])
                if k not in first[j] and len(second[j]) < args.ref_block:
                    second[j].append(k)
        ref2_w = {}
        for j, ks in second.items():
            if len(ks) < args.ref_block:
                continue
            m = sum(rewards[k] for k in ks) / len(ks)
            for k in ks:
                if rewards[k] != m:
                    ref2_w[k] = rewards[k] - m
        runner.model.zero_grad(set_to_none=True)
        backward_weighted(runner, args, answers, sorted(ref2_w), ref2_w, norm)
        ref2 = runner.grad_snapshot()
    runner.model.zero_grad(set_to_none=True)
    live_a = [k for k, a in enumerate(adv) if a != 0 and owner[k] in half_a]
    live_b = [k for k, a in enumerate(adv) if a != 0 and owner[k] not in half_a]
    backward_weighted(runner, args, answers, live_a, adv, norm, entropy)
    ga = runner.grad_snapshot()
    backward_weighted(runner, args, answers, live_b, adv, norm, entropy)
    stats = dict.fromkeys(("aa", "bb", "ab", "ar", "br", "rr"), 0.0)
    if ref2 is not None:
        stats.update(dict.fromkeys(("r1r2", "r2r2", "ar2"), 0.0))
    with torch.no_grad():
        for i, (p, a_cpu, r_cpu) in enumerate(zip(runner.params, ga, ref)):
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            a = a_cpu.to(runner.device).flatten()
            r = r_cpu.to(runner.device).flatten()
            b = g.flatten() - a
            stats["aa"] += float(torch.dot(a, a))
            stats["bb"] += float(torch.dot(b, b))
            stats["ab"] += float(torch.dot(a, b))
            stats["ar"] += float(torch.dot(a, r))
            stats["br"] += float(torch.dot(b, r))
            stats["rr"] += float(torch.dot(r, r))
            if ref2 is not None:
                r2 = ref2[i].to(runner.device).flatten()
                stats["r1r2"] += float(torch.dot(r, r2))
                stats["r2r2"] += float(torch.dot(r2, r2))
                stats["ar2"] += float(torch.dot(a, r2))
    stats.update({"prompts_a": len(half_a), "prompts_b": len(prompts) - len(half_a),
                  "answers_a": sum(1 for j in owner if j in half_a),
                  "answers_b": sum(1 for j in owner if j not in half_a),
                  "ref_answers": len(ref_w), "norm": norm})
    return stats, len(live_a) + len(live_b)


def train_arm(args, arm, w, problems, eval_set, out_dir, calib=None, final_set=None):
    runner = Runner(args)
    opt = torch.optim.AdamW(runner.params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    rng = random.Random(args.seed)
    order = list(range(len(problems)))
    rng.shuffle(order)
    log, evals, sampled = [], [], []
    norm = args.budget * args.max_new
    calib_groups = (calib or {}).get("groups") or [[1, 8]]
    ratio = (calib["F0"] / calib["F1"]) if calib and calib.get("F1") else 1.0 / w
    spent, prompts_done, pointer, mean_n, recent = 0, 0, 0, float(args.group), []
    eval_steps = ({int(s) for s in args.eval_steps.split(",")} if args.eval_steps
                  else set(range(0, args.steps + 1, args.eval_every)))
    sample_steps = {int(s) for s in args.sample_eval_steps.split(",") if s}
    passk_steps = {int(s) for s in args.passk_steps.split(",") if s}
    surrogate = args.reuse > 1 or args.length_norm or args.norm_selected
    assert not (surrogate and args.diagnostics), "diagnostics need the plain on-policy loss"
    assert not surrogate or arm in FIXED_ARMS, "verl-like options are implemented for fixed groups only"
    passk = []
    for step in range(args.steps + 1):
        if step in eval_steps:
            acc, per = runner.evaluate(eval_set, args.seed)
            evals.append({"step": step, "accuracy": acc, "correct": per})
            print(f"[{arm}] step {step} eval accuracy {acc:.4f}", flush=True)
            (out_dir / f"{arm}_eval.json").write_text(json.dumps(evals))
        if step in sample_steps:
            acc, per = runner.evaluate(eval_set, args.seed + 17 * step, k=args.sample_k, greedy=False)
            sampled.append({"step": step, "accuracy": acc, "k": args.sample_k, "correct": per})
            print(f"[{arm}] step {step} sampled accuracy {acc:.4f} (k={args.sample_k})", flush=True)
            (out_dir / f"{arm}_sampled_eval.json").write_text(json.dumps(sampled))
        if step in passk_steps:
            _, per = runner.evaluate(eval_set[: args.passk_limit], args.seed + 31 * step, k=args.passk_n, greedy=False)
            curve = {k: sum(pass_at_k(args.passk_n, c, k) for c in per) / len(per)
                     for k in (1, 2, 4, 8, 16, 32, 64) if k <= args.passk_n}
            passk.append({"step": step, "n": args.passk_n, "counts": per, "pass_at_k": curve})
            print(f"[{arm}] step {step} pass@k " + " ".join(f"{k}:{v:.3f}" for k, v in curve.items()), flush=True)
            (out_dir / f"{arm}_passk.json").write_text(json.dumps(passk))
        if step == args.steps:
            if args.save_model:
                path = Path(args.save_model.replace("{arm}", arm))
                bf16 = {k: v.detach().to(torch.bfloat16) for k, v in runner.model.state_dict().items()}
                runner.model.save_pretrained(path, state_dict=bf16, safe_serialization=True)
                runner.tok.save_pretrained(path)
                print(f"[{arm}] saved model to {path}", flush=True)
                del bf16
                if runner.device == "cuda":
                    torch.cuda.empty_cache()
            if final_set:
                acc, per = runner.evaluate(final_set, args.seed)
                print(f"[{arm}] final held-out accuracy {acc:.4f} over {len(final_set)} problems", flush=True)
                (out_dir / f"{arm}_final_eval.json").write_text(
                    json.dumps({"step": step, "accuracy": acc, "n": len(final_set), "correct": per}))
            break
        if runner.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        extra = {}
        if arm in FIXED_ARMS:
            pps, G = args.prompts_per_step, args.group
            if step % args.reuse == 0:
                n_pool = pps * args.reuse
                batch = [problems[order[(step * pps + j) % len(order)]] for j in range(n_pool)]
                texts = [build_prompt(runner.tok, q["question"]) for q in batch]
                pool_answers = runner.generate(texts, G, args.max_new, args.seed * 100003 + step)
                pool_rewards = [reward(runner.tok.decode(r), batch[k // G]["answer"]) for k, (_, r) in enumerate(pool_answers)]
                pool_old = old_logprobs(runner, args, pool_answers) if surrogate else None
            j0 = (step % args.reuse) * pps * G
            answers = pool_answers[j0:j0 + pps * G]
            rewards = pool_rewards[j0:j0 + pps * G]
            old_lp = pool_old[j0:j0 + pps * G] if surrogate else None
            adv = []
            for j in range(pps):
                g = rewards[j * G:(j + 1) * G]
                m = sum(g) / len(g)
                for r in g:
                    adv.append(fixed_advantage(arm, r, m, w, args.pos_weight))
            groups = [[sum(rewards[j * G:(j + 1) * G]), G] for j in range(pps)]
            extra["stale"] = step % args.reuse
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
        owner = [j for j, (_, n) in enumerate(groups) for _ in range(n)]
        entropy = []
        if surrogate:
            live = [k for k, a in enumerate(adv) if a != 0]
            step_norm = (len(live) * args.max_new if args.norm_selected else norm)
            if live:
                backward_surrogate(runner, args, answers, live, adv, old_lp, step_norm)
            n_live = len(live)
        elif args.diagnostics:
            stats, n_live = diagnostic_step(runner, args, answers, adv, rewards, owner, norm, step, entropy)
            extra["split"] = stats
        else:
            live = [k for k, a in enumerate(adv) if a != 0]
            backward_weighted(runner, args, answers, live, adv, norm, entropy)
            n_live = len(live)
        live_tokens = sum(len(answers[k][1]) for k, a in enumerate(adv) if a != 0)
        if args.entropy_every and step % args.entropy_every == 0:
            extra["probe"] = probe_entropy(runner, args, answers, rewards, owner)
        seen, first_block = {}, []
        for k, j in enumerate(owner):
            seen[j] = seen.get(j, 0) + 1
            if seen[j] <= args.ref_block:
                first_block.append(rewards[k])
        grad_norm = float(torch.nn.utils.clip_grad_norm_(runner.params, args.clip)) if n_live else 0.0
        if n_live:
            opt.step()
        opt.zero_grad(set_to_none=True)
        mixed = sum(0 < k < n for k, n in groups)
        rec = {"step": step, "mean_reward": sum(rewards) / len(rewards),
               "first_block_reward": sum(first_block) / len(first_block), "mixed_groups": mixed,
               "groups": len(groups), "grad_norm": grad_norm, "answers": len(answers),
               "tokens": sum(len(r) for _, r in answers),
               "mean_length": sum(len(r) for _, r in answers) / len(answers),
               "entropy": sum(entropy) / live_tokens if live_tokens else None,
               "group_counts": groups,
               "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if runner.device == "cuda" else None,
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
    parser.add_argument("--prefill-tokens", type=int, default=256,
                        help="shrink a generation batch so batch x padded prompt length stays under gen-batch x this")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-steps", default=None, help="comma-separated greedy evaluation steps (overrides --eval-every)")
    parser.add_argument("--sample-eval-steps", default="", help="steps at which to measure sampled accuracy")
    parser.add_argument("--sample-k", type=int, default=4, help="answers per problem for sampled accuracy")
    parser.add_argument("--final-eval", default=None, help="larger held-out set, evaluated greedily after the last step")
    parser.add_argument("--final-limit", type=int, default=None)
    parser.add_argument("--pos-weight", type=float, default=0.1, help="weight of correct answers in wreinforce")
    parser.add_argument("--reuse", type=int, default=1, help="updates per rollout batch (stale samples after the first)")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio for the surrogate loss")
    parser.add_argument("--length-norm", action="store_true", help="average each answer's loss over its tokens")
    parser.add_argument("--norm-selected", action="store_true",
                        help="normalize the loss by the number of answers with nonzero advantage in the update")
    parser.add_argument("--entropy-every", type=int, default=0, help="probe entropy on first-block answers every k steps")
    parser.add_argument("--passk-steps", default="", help="steps at which to measure pass@k")
    parser.add_argument("--passk-n", type=int, default=64, help="samples per problem for pass@k")
    parser.add_argument("--passk-limit", type=int, default=None, help="evaluate pass@k on the first problems only")
    parser.add_argument("--save-model", default=None, help="directory for the final model ({arm} is replaced)")
    parser.add_argument("--diagnostics", action="store_true", help="split-half gradient statistics every step")
    parser.add_argument("--ref-block", type=int, default=4)
    parser.add_argument("--within-ref", action="store_true",
                        help="second independent reference within prompts (measures per-prompt signal)")
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
    args, unknown = parser.parse_known_args()
    if unknown:
        print("ignoring unknown arguments:", unknown, flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.budget is None:
        args.budget = args.prompts_per_step * args.group
    problems = json.loads(Path(args.train).read_text())
    eval_set = json.loads(Path(args.eval).read_text())[: args.eval_limit]
    final_set = None
    if args.final_eval:
        path = Path(args.final_eval) if Path(args.final_eval).exists() else HERE / "data" / args.final_eval
        final_set = json.loads(path.read_text())[: args.final_limit]
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
        train_arm(args, arm, w, problems[: -args.calib_prompts], eval_set, out_dir, calib, final_set)
    print("done", flush=True)


if __name__ == "__main__":
    main()
