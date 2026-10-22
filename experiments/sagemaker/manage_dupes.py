"""Keep one copy per key among duplicate jobs: the first to start; stop the other copies that have not started.

Usage: python manage_dupes.py JOBS_FILE KEY_REGEX
JOBS_FILE lists region:job lines (re-read every 2 minutes); KEY_REGEX's groups form the key, e.g.
'worth-v14e-(E\\dL\\d)-s(\\d+)-' for evaluation lanes. Exits when every key has a started copy or none left running.
"""

import re
import sys
import time

from manage_lanes import HERE, started, state, stop


def main():
    path, pattern = HERE / sys.argv[1], re.compile(sys.argv[2])
    while True:
        groups = {}
        for spec in path.read_text().split():
            region, job = spec.split(":", 1)
            m = pattern.search(job)
            if m:
                groups.setdefault(m.groups(), []).append((region, job))
        unresolved = 0
        for key, copies in sorted(groups.items()):
            sts = [(r, j, state(r, j)) for r, j in copies]
            live = [x for x in sts if x[2][0] in ("InProgress", "Completed")]
            first = [x for x in live if started(x[2])]
            if first:
                for r, j, st in live:
                    if st[0] == "InProgress" and not started(st):
                        stop(r, j)
                        print(time.strftime("%H:%M:%S"), f"{key}: kept {first[0][1]}", flush=True)
            elif live:
                unresolved += 1
        if unresolved == 0:
            break
        time.sleep(120)
    print("all resolved", flush=True)


if __name__ == "__main__":
    main()
