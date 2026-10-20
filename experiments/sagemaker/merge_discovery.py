"""Merge Pre-registration 14's discovery shards into training/data/discovery_qwen_train32.json.

Usage: AWS_PROFILE=stop-ro python merge_discovery.py [--shards 8]
Reads the jobs in v14d_jobs.txt, takes each shard from the first job whose live copy is complete,
and writes the merged archive in training-file order only when every shard is complete.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "training" / "data"
ACCOUNT = "165919938845"


def listing(region, job):
    """Discovery shard files present for a job (live checkpoint copy and final output)."""
    out = {}
    for prefix in (f"checkpoints/{job}/", f"output/{job}/output/"):
        r = subprocess.run(["aws", "s3", "ls", "--recursive", "--region", region, f"s3://stop-rl-{ACCOUNT}-{region}/{prefix}"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            key = line.split()[-1]
            if "discovery_shard" in key:
                out[int(key.rsplit("discovery_shard", 1)[1].split(".")[0])] = f"s3://stop-rl-{ACCOUNT}-{region}/{key}"
    return out


def fetch(region, uri, tmp):
    dest = Path(tmp) / uri.replace("/", "_")
    if subprocess.run(["aws", "s3", "cp", "--only-show-errors", "--region", region, uri, str(dest)], capture_output=True).returncode:
        return None
    return json.loads(dest.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8)
    args = ap.parse_args()
    train = json.loads((DATA / "train.json").read_text())
    jobs = [line.strip().split(":", 1) for line in (HERE / "v14d_jobs.txt").read_text().splitlines() if line.strip()]
    best = {}
    files = {(region, job): listing(region, job) for region, job in jobs}
    with tempfile.TemporaryDirectory() as tmp:
        for shard in range(args.shards):
            want = train[shard::args.shards]
            for region, job in jobs:
                uri = files[(region, job)].get(shard)
                d = fetch(region, uri, tmp) if uri else None
                if d is None:
                    continue
                if len(d) > len(best.get(shard, (None, []))[1]):
                    best[shard] = (job, d)
                if len(d) == len(want):
                    break
            got = len(best.get(shard, (None, []))[1])
            print(f"shard {shard}: {got}/{len(want)}" + (f" from {best[shard][0]}" if shard in best else ""))
    if any(len(best.get(s, (None, []))[1]) != len(train[s::args.shards]) for s in range(args.shards)):
        print("incomplete; nothing written")
        return
    merged = []
    for i, q in enumerate(train):
        rec = best[i % args.shards][1][i // args.shards]
        assert rec["question"] == q["question"] and rec["answer"] == q["answer"], i
        merged.append(rec)
    assert len({r["question"] for r in merged}) == len(merged)
    out = DATA / "discovery_qwen_train32.json"
    out.write_text(json.dumps(merged))
    solved = sum(r["response"] is not None for r in merged)
    c = [r["c"] for r in merged]
    print(f"wrote {out}: {len(merged)} questions, {solved} with a retained success, "
          f"{sum(x > 0 for x in c)} solved at least once, mean success rate {sum(c) / (32 * len(c)):.3f}")
    print("sources: " + ", ".join(f"{s}: {best[s][0]}" for s in sorted(best)))


if __name__ == "__main__":
    main()
