"""Binary answer verifier: last \\boxed{...} in the response against a reference."""

import re
from fractions import Fraction


def last_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 1, start + len("\\boxed{")
    begin = i
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[begin:i - 1] if depth == 0 else None


def normalize(answer):
    s = answer.strip().strip("$").replace(" ", "").replace("\\!", "").replace("\\,", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = s.replace("\\left", "").replace("\\right", "").rstrip(".")
    s = re.sub(r"^[a-zA-Z]=", "", s)
    return s


def as_number(s):
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?(?:/\(?(-?\d+(?:\.\d+)?)\)?)?", s)
    if not m:
        return None
    value = Fraction(m.group(1))
    if not m.group(2):
        return value
    denominator = Fraction(m.group(2))
    return value / denominator if denominator != 0 else None


def reward(response, reference):
    boxed = last_boxed(response)
    if boxed is None:
        return 0
    got, want = normalize(boxed), normalize(reference)
    if got == want:
        return 1
    a, b = as_number(got), as_number(want)
    return int(a is not None and b is not None and a == b)


if __name__ == "__main__":
    cases = [
        ("so \\boxed{1358}.", "1358", 1),
        ("area \\boxed{\\frac{175}{2}}", "175/2", 1),
        ("area \\boxed{87.5}", "175/2", 1),
        ("\\boxed{\\dfrac{35}{66}}", "35/66", 1),
        ("\\boxed{12} then later \\boxed{13}", "12", 0),
        ("no box here: 5", "5", 0),
        ("\\boxed{x=5}", "5", 1),
        ("\\boxed{5.}", "5", 1),
        ("\\boxed{\\text{34}}", "34", 1),
        ("\\boxed{1/0}", "5", 0),
        ("\\boxed{\\frac{3}{0}}", "5", 0),
    ]
    for text, ref, expected in cases:
        assert reward(text, ref) == expected, (text, ref, expected)
    print(f"verifier self-test: {len(cases)} cases passed")
