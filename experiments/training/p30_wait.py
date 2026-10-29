"""Wait (polling) until every given start directory has its start_meta.json, so a lane can evaluate a start that another
lane trains. Fails after --timeout seconds. Usage: python p30_wait.py --dirs D1 D2 ... --out DIR [--model ignored]"""
import argparse
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--timeout", type=int, default=7200)
ap.add_argument("--out", default=None)
ap.add_argument("--model", default=None)
a = ap.parse_args()
t0 = time.time()
while not all((Path(d) / "start_meta.json").exists() for d in a.dirs):
    if time.time() - t0 > a.timeout:
        raise SystemExit(f"timed out waiting for {[d for d in a.dirs if not (Path(d) / 'start_meta.json').exists()]}")
    time.sleep(20)
time.sleep(30)  # let the writer finish the tokenizer files saved alongside
print("ready:", a.dirs, flush=True)
