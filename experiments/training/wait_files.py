"""Wait (polling) until every given file exists, then a grace period (so a lane can start after another lane's run has
saved its last checkpoint and exited). Fails after --timeout seconds. Usage: python wait_files.py --files F1 F2 ...
[--grace 180] [--timeout 14400] (--out and --model are accepted and ignored, for entry.py's script steps)"""
import argparse
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--grace", type=int, default=180)
ap.add_argument("--timeout", type=int, default=14400)
ap.add_argument("--out", default=None)
ap.add_argument("--model", default=None)
a = ap.parse_args()
t0 = time.time()
while not all(Path(f).exists() for f in a.files):
    if time.time() - t0 > a.timeout:
        raise SystemExit(f"timed out waiting for {[f for f in a.files if not Path(f).exists()]}")
    time.sleep(30)
time.sleep(a.grace)
print("ready:", a.files, flush=True)
