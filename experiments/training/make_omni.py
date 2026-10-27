"""Fresh hard benchmark outside MATH (frozen before any model is evaluated on it): Omni-MATH (KbsdJames/Omni-MATH,
test.jsonl, downloaded to data/raw_omni/) problems whose reference answer is a number our checker parses (integer,
decimal or fraction), difficulty >= 4, not in MATH (exact question text), a random 512 in a fixed order (seed 2029).
Writes data/omni_hard_512.json with question, answer, level (difficulty), subject (domain), source.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verifier import as_number, normalize  # noqa: E402


def main():
    rows = [json.loads(l) for l in open(HERE / "data" / "raw_omni" / "test.jsonl")]
    math_q = {q["question"] for f in ("eval_math_test_numeric.json", "train.json") for q in json.loads((HERE / "data" / f).read_text())}
    pool = [r for r in rows if as_number(normalize(str(r["answer"]))) is not None and float(r["difficulty"]) >= 4 and r["problem"] not in math_q]
    pool.sort(key=lambda r: r["problem"])
    random.Random(2029).shuffle(pool)
    out = [{"question": r["problem"], "answer": str(r["answer"]), "level": float(r["difficulty"]),
            "subject": r["domain"][0] if r["domain"] else "", "source": r["source"]} for r in pool[:512]]
    (HERE / "data" / "omni_hard_512.json").write_text(json.dumps(out, indent=0))
    print(f"pool {len(pool)}, written {len(out)}")


if __name__ == "__main__":
    main()
