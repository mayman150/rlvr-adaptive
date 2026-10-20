"""Keep either a 4-lane evaluation job or its four single-lane copies, whichever starts first.

Usage: python manage_lanes.py lanes_launch.txt
Every 2 minutes: for each multi-lane job (v12e-E<k>-s<seed>) still pending, if one of its single
lanes (v12e-E<k>L<i>-s<seed>) has started, stop the multi-lane job; if the multi-lane job has
started, stop its single lanes that have not started. Exits when every pair is resolved.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACCOUNT = "165919938845"


def aws(args, profile):
    r = subprocess.run(["aws", "--profile", profile, *args], capture_output=True, text=True)
    if "ExpiredToken" in r.stderr:
        role = "ReadOnly" if profile == "stop-ro" else "Admin"
        subprocess.run(["ada", "credentials", "update", f"--account={ACCOUNT}", "--provider=isengard", f"--role={role}",
                        f"--profile={profile}", "--once"], capture_output=True)
        r = subprocess.run(["aws", "--profile", profile, *args], capture_output=True, text=True)
    return r.stdout.strip()


def state(region, job):
    out = aws(["--region", region, "sagemaker", "describe-training-job", "--training-job-name", job,
               "--query", "[TrainingJobStatus,SecondaryStatus]", "--output", "text"], "stop-ro")
    return out.split() if out else ["?", "?"]


def started(st):
    return st[0] == "Completed" or (st[0] == "InProgress" and st[1] not in ("Pending", "Starting"))


def stop(region, job):
    aws(["--region", region, "sagemaker", "stop-training-job", "--training-job-name", job], "stop-admin")
    print(time.strftime("%H:%M:%S"), "stopped", job, flush=True)


def main():
    jobs = [l.strip() for l in (HERE / "v12e_jobs.txt").read_text().splitlines() if l.strip()]
    singles = {}
    for line in Path(sys.argv[1]).read_text().splitlines():
        m = re.search(r"started (worth-v12e-(E\d)L(\d)-s(\d+)-\S+)", line)
        if m:
            singles.setdefault((m.group(2), m.group(4)), []).append(m.group(1))
    region_of = {"8": "us-west-2", "9": "us-east-1", "10": "us-east-2"}
    multi = {}
    for spec in jobs:
        r, j = spec.split(":")
        m = re.match(r"worth-v12e-(E\d)-s(\d+)-", j)
        if m and (m.group(1), m.group(2)) in singles:
            multi.setdefault((m.group(1), m.group(2)), []).append((r, j))
    pending = set(singles)
    while pending:
        for key in sorted(pending):
            e, seed = key
            ms = [(r, j, state(r, j)) for r, j in multi.get(key, [])]
            ms = [x for x in ms if x[2][0] == "InProgress" or x[2][0] == "Completed"]
            ss = [(region_of[seed], j, state(region_of[seed], j)) for j in singles[key]]
            if any(started(st) for _, _, st in ms):
                for r, j, st in ss:
                    if st[0] == "InProgress" and not started(st):
                        stop(r, j)
                if all(started(st) or st[0] != "InProgress" for _, _, st in ss):
                    print(time.strftime("%H:%M:%S"), f"{key}: multi-lane job kept", flush=True)
                    pending.discard(key)
            elif any(started(st) for _, _, st in ss):
                for r, j, st in ms:
                    if st[0] == "InProgress":
                        stop(r, j)
                print(time.strftime("%H:%M:%S"), f"{key}: single lanes kept", flush=True)
                pending.discard(key)
            elif not ms:
                pending.discard(key)  # no multi-lane job left: singles only
        time.sleep(120)
    print("all resolved", flush=True)


if __name__ == "__main__":
    main()
