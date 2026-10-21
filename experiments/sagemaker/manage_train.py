"""Keep either a seed's 4-lane Pre-registration 14 training job or its single-lane copies, whichever starts first.

Usage: python manage_train.py   (reads v14_jobs.txt for worth-v14-ALL-s<seed> and v14_singles.txt)
Every 2 minutes, per seed: if the 4-lane job has started, stop the seed's single lanes; if any single
lane has started first, stop the 4-lane job and keep the single lanes. Exits when every seed is resolved.
"""

import re
import time

from manage_lanes import HERE, started, state, stop


def main():
    multi = {}
    for spec in (HERE / "v14_jobs.txt").read_text().split():
        m = re.match(r"(\S+):(worth-v14-ALL-s(\d+)-\S+)", spec)
        if m:
            multi[m.group(3)] = (m.group(1), m.group(2))
    singles = {}
    for spec in (HERE / "v14_singles.txt").read_text().split():
        m = re.match(r"(\S+):(worth-v14-\w+?-s(\d+)-\S+)", spec)
        if m:
            singles.setdefault(m.group(3), []).append((m.group(1), m.group(2)))
    pending = set(multi) & set(singles)
    while pending:
        for seed in sorted(pending):
            r, j = multi[seed]
            mst = state(r, j)
            ss = [(sr, sj, state(sr, sj)) for sr, sj in singles[seed]]
            if started(mst):
                for sr, sj, st in ss:
                    if st[0] == "InProgress":
                        stop(sr, sj)
                print(time.strftime("%H:%M:%S"), f"seed {seed}: 4-lane job kept", flush=True)
                pending.discard(seed)
            elif any(started(st) for _, _, st in ss):
                if mst[0] == "InProgress":
                    stop(r, j)
                print(time.strftime("%H:%M:%S"), f"seed {seed}: single lanes kept", flush=True)
                pending.discard(seed)
            elif mst[0] != "InProgress":
                pending.discard(seed)
        time.sleep(120)
    print("all resolved", flush=True)


if __name__ == "__main__":
    main()
