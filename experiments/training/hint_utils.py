"""Answer-safe hint prefixes for the hint curriculum (Pre-registration 40).

A hint is a prefix of a solution's lines. The prefix never shows the answer: the trace is cut before its FIRST box, and
the lines stop before the first line that reveals the reference answer. Lines are normalized first (LaTeX delimiters,
spacing, \\frac / \\dfrac / \\tfrac including the brace-free \\frac52 form, a/b, decimals), then:
- a numeric answer is revealed by any number in the line with the same value (integers, decimals such as 24.0,
  fractions); for small integers (|v| <= 10), which are common in set-up lines, only in a conclusion form (after =, is,
  equals, answer, find/get/obtain/gives/yields/total/are, ':' or at the line end);
- a non-numeric answer is revealed by its normalized string anywhere in the line.
"""
import re
from fractions import Fraction

from verifier import as_number, normalize

_FRAC = re.compile(r"\\[dtc]?frac\s*(?:\{\s*(-?[\d.]+)\s*\}|(\d))\s*(?:\{\s*(-?[\d.]+)\s*\}|(\d))")
_NUM = re.compile(r"(?<![A-Za-z\d.])(-?\d+(?:\.\d+)?)(?:\s*/\s*(-?\d+(?:\.\d+)?))?(?![A-Za-z\d]|\.\d)")
PROSE_BEFORE = re.compile(r"(?:\bthere (?:are|is|were)\b|\bwe (?:have|get|find|obtain)\b|\bexactly\b|\btotal of\b|\bin total\b|\baltogether\b|\bhence\b|\btherefore\b|\bthus\b|\bso\b)[^.;]{0,25}$", re.I)
PROSE_AFTER = re.compile(r"^\s*(?:possible|distinct|different|ways|values|solutions|outcomes|choices|arrangements|numbers|integers|pairs|cases)\b", re.I)
LEAD = re.compile(r"(?:=|\bis\b|\bequals\b|answer|\bfind\b|\bget\b|\bobtain\b|\bgives?\b|\byields?\b|\btotal\b|\bare\b|:)\s*$")


def norm_line(line):
    s = line.replace("\\(", " ").replace("\\)", " ").replace("\\[", " ").replace("\\]", " ").replace("$", " ")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", " ")
    s = _FRAC.sub(lambda m: f"{m.group(1) or m.group(2)}/{m.group(3) or m.group(4)}", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # thousands separators: 15,625 -> 15625
    s = re.sub(r"_\{?\s*(\d+)\s*\}?", r" base\1 ", s)  # base subscripts: 60_{10} -> 60 base10
    return re.sub(r"\s+", " ", s)


def _value(num, den):
    try:
        v = Fraction(num)
        return v / Fraction(den) if den is not None and Fraction(den) != 0 else (v if den is None else None)
    except (ValueError, ZeroDivisionError):
        return None


def reveals(line, answer):
    s = norm_line(line)
    target = as_number(normalize(answer.strip().strip("$")))
    if target is None:
        strip = lambda x: re.sub(r"[\\\s{}]", "", normalize(x))
        a = strip(answer)
        return bool(a) and a in strip(s)
    small = target.denominator == 1 and abs(target.numerator) <= 10
    for m in _NUM.finditer(s):
        if _value(m.group(1), m.group(2)) != target:
            continue
        if not small:
            return True
        before, after = s[:m.start()], s[m.end():]
        if LEAD.search(before) or PROSE_BEFORE.search(before) or PROSE_AFTER.search(after) or re.fullmatch(r"\s*\.?\s*", after):
            return True
    return False


def safe_hint_lines(trace, answer):
    cut = trace.find("\\boxed")
    pre = trace[:cut] if cut >= 0 else trace
    out = []
    for line in pre.split("\n"):
        if not line.strip():
            continue
        if reveals(line, answer):
            break
        out.append(line)
    return out
