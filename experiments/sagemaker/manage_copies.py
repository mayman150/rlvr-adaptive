"""Keep one copy per (arm, seed) of Pre-registration 14 training: the first to start.

Usage: python manage_copies.py   (reads v14_singles.txt and v14_jobs.txt, re-read every cycle)
Every 2 minutes: if a seed's 4-lane job (worth-v14-ALL-s<seed>) has started, stop all of that seed's
single-lane copies still running; otherwise, once one copy of an (arm, seed) has started, stop that
pair's other copies that have not started. Exits when every pair has one started copy or none left.
"""

import re
import time

from manage_lanes import HERE, started, state, stop


def specs(name, pattern):
    out = []
    for spec in (HERE / name).read_text().split():
        m = re.match(pattern, spec)
        if m:
            out.append(m)
    return out


def main():
    while True:
        multi = {m.group(3): (m.group(1), m.group(2)) for m in specs("v14_jobs.txt", r"(\S+):(worth-v14-ALL-s(\d+)-\S+)")}
        pairs = {}
        for m in specs("v14_singles.txt", r"(\S+):(worth-v14-(\w+?)-s(\d+)-\S+)"):
            pairs.setdefault((m.group(3), m.group(4)), []).append((m.group(1), m.group(2)))
        unresolved = 0
        for (arm, seed), copies in sorted(pairs.items()):
            sts = [(r, j, state(r, j)) for r, j in copies]
            live = [x for x in sts if x[2][0] in ("InProgress", "Completed")]
            if seed in multi and started(state(*multi[seed])):
                for r, j, st in live:
                    if st[0] == "InProgress":
                        stop(r, j)
                continue
            first = [x for x in live if started(x[2])]
            if first:
                for r, j, st in live:
                    if st[0] == "InProgress" and not started(st):
                        stop(r, j)
                        print(time.strftime("%H:%M:%S"), f"{arm} seed {seed}: kept {first[0][1]}", flush=True)
            elif live:
                unresolved += 1
        if unresolved == 0:
            break
        time.sleep(120)
    print("all resolved", flush=True)


if __name__ == "__main__":
    main()
