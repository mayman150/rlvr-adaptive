"""Answer-safe hint prefixes for the hint curriculum (Pre-registration 40).

A hint is a prefix of a solution's lines. The prefix never shows the answer: the trace is cut before its FIRST box, and
the lines stop before the first line that states the reference answer as a conclusion (after "=", "is", "equals",
"answer", or ":" or at the end of the line), in its raw, \\dfrac/\\frac or decimal form.
"""
import re

from verifier import as_number, normalize


def _forms(answer):
    a = answer.strip().strip("$")
    forms = {a, a.replace("\\dfrac", "\\frac"), a.replace("\\frac", "\\dfrac")}
    v = as_number(normalize(a))
    if v is not None:
        if v.denominator == 1:
            forms.add(str(v.numerator))
        else:
            forms.add(f"{v.numerator}/{v.denominator}")
            forms.add(f"\\frac{{{v.numerator}}}{{{v.denominator}}}")
            forms.add(f"\\dfrac{{{v.numerator}}}{{{v.denominator}}}")
            d = float(v)
            forms.add(f"{d:.4g}")
    return [f for f in forms if f]


def answer_conclusion(answer):
    """Lines that reveal the answer. Small integers (|v| <= 10) are common in set-up lines, so for them only a
    conclusion form counts (after =, is, equals, answer, find/get/obtain/gives/yields/total/are, ':' or at the line
    end); any other answer (larger integers, fractions, expressions) counts wherever it appears as a token."""
    forms = sorted(_forms(answer), key=len, reverse=True)
    alts = "|".join(re.escape(f) for f in forms)
    v = as_number(normalize(answer.strip().strip("$")))
    small = v is not None and v.denominator == 1 and abs(v.numerator) <= 10
    tok = rf"(?<![\w.\d])(?:{alts})(?![\w\d]|\.\d)"
    if not small:
        return re.compile(tok)
    lead = r"(?:=|\bis\b|\bequals\b|answer|\bfind\b|\bget\b|\bobtain\b|\bgives?\b|\byields?\b|\btotal\b|\bare\b|:)"
    return re.compile(rf"{lead}\s*\$?\s*(?:{alts})(?![\w\d]|\.\d)|{tok}\s*\$?\s*\.?\s*\$?\s*$")


def safe_hint_lines(trace, answer):
    cut = trace.find("\\boxed")
    pre = trace[:cut] if cut >= 0 else trace
    pat = answer_conclusion(answer)
    out = []
    for line in pre.split("\n"):
        if not line.strip():
            continue
        if pat.search(line):
            break
        out.append(line)
    return out
