"""Write <out>/DONE after the preceding step of a lane has exited (successfully or not), recording whether the
given files exist (for example a run's last checkpoint). Lets other lanes wait for the end of a multi-GPU run, after
its GPUs are released, without failing open. Usage: python mark_done.py --check F1 F2 ... --out DIR [--model ignored]"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--check", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default=None)
a = ap.parse_args()
ok = all(Path(f).exists() for f in a.check)
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / "DONE").write_text(json.dumps({"ok": ok, "checked": a.check}))
print(f"done marker: ok={ok}", flush=True)
