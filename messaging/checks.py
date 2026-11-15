"""Deterministic message checks. Each returns (passed, note)."""
import re

GREETINGS = {"en": ("hi", "hello", "dear"), "fr": ("bonjour", "salut", "cher", "chère")}
MAX_CHARS = {"sms": 320, "push": 180, "in_app": 400, "email": 1200, "phone": 800}
SIGNATURE = "Claire"


def lines(msg):
    return [l.strip() for l in msg.strip().splitlines() if l.strip()]


def currency(msg, facts):
    bad = re.findall(r"\b(?:EUR|USD|euros?)\b|\$", msg, re.I)
    return not bad, f"use €, found {sorted(set(bad))}" if bad else "ok"


def greeting(msg, facts):
    first = (lines(msg) or [""])[0].lower()
    ok = first.startswith(GREETINGS[facts["language"]])
    return ok, "ok" if ok else f"first line {first[:40]!r} is not a {facts['language_name']} greeting"


def signature(msg, facts):
    last = (lines(msg) or [""])[-1]
    ok = SIGNATURE in last
    return ok, "ok" if ok else f"last line {last[:40]!r} is not signed {SIGNATURE}"


def parse_numbers(text):
    """Numbers in English (1,234.50) or French (1 234,50) notation."""
    text = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}\b)", "", text)  # French thousands spaces
    out = []
    for s in re.findall(r"\d+(?:[.,]\d+)*", text):
        if "," in s and "." in s:
            dec = max(s.rfind(","), s.rfind("."))
            s = re.sub(r"[.,]", "", s[:dec]) + "." + s[dec + 1:]
        else:
            sep = "," if "," in s else "." if "." in s else None
            if sep:  # a trailing group of exactly 3 digits is thousands, otherwise a decimal part
                head, tail = s.rsplit(sep, 1)
                s = head.replace(sep, "") + ("" if len(tail) == 3 else ".") + tail
        out.append(float(s))
    return out


def allowed_numbers(facts_text):
    allowed = []
    for line in facts_text.splitlines():
        nums = parse_numbers(line)
        allowed += nums + ([n * 12 for n in nums] if "month" in line else [])
    return allowed


def numbers(msg, facts):
    allowed = allowed_numbers(facts["facts"])
    invented = [n for n in parse_numbers(msg) if not any(abs(n - a) <= max(0.5, 0.01 * a) for a in allowed)]
    return not invented, f"not in the facts: {invented}" if invented else "ok"


def length(msg, facts):
    limit = MAX_CHARS[facts["channel"]]
    return len(msg) <= limit, f"{len(msg)}/{limit} chars"


CHECKS = {"currency": currency, "greeting": greeting, "signature": signature, "numbers": numbers, "length": length}


def run(msg, facts):
    return {name: check(msg, facts) for name, check in CHECKS.items()}
