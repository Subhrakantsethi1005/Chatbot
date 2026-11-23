"""Text normalisation shared by ingestion, duplicate detection, retrieval and
security checks."""

from __future__ import annotations

import re
import unicodedata

# Zero-width, bidi-override and other invisible characters that are commonly
# used to hide instructions inside otherwise normal-looking text.
INVISIBLE_RE = re.compile(
    "[­͏؜ᅟᅠ឴឵᠋-᠎​-‏"
    "‪-‮⁠-⁯ㅤ︀-️﻿ﾠ]"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_APOSTROPHE_RE = re.compile(r"(?<=\w)['’`](?=\w)")
_SPACE_RE = re.compile(r"[ \t]+")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself nor of off on once only or
    other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your
    yours yourself yourselves hi hello hey please guys anyone someone could would also
    get got im ive id dont""".split()
)


def clean_text(text: str) -> str:
    """Unicode-normalise, drop invisible/control characters, unify newlines."""
    text = unicodedata.normalize("NFKC", text or "")
    text = INVISIBLE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    # collapse runs of blank lines
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def normalize(text: str) -> str:
    """Lower-case, punctuation-free, single-spaced form used for comparisons."""
    text = unicodedata.normalize("NFKC", text or "")
    text = INVISIBLE_RE.sub("", text).lower()
    text = _APOSTROPHE_RE.sub("", text)  # "don't" -> "dont"
    text = text.replace("_", " ")
    text = _NON_WORD_RE.sub(" ", text)
    return " ".join(text.split())


def stem(token: str) -> str:
    """A deliberately small suffix stripper (good enough for FAQ retrieval)."""
    if len(token) <= 4:
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith(("sses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


# Map words that mean the same thing in a course-support context onto one
# token so "coding" finds "programming" and "how long" finds "duration".
CANONICAL = {
    "code": "program", "coding": "program", "coder": "program", "programm": "program", "programmer": "program",
    "requirement": "prerequisite", "require": "prerequisite", "requir": "prerequisite",
    "duration": "long", "fee": "price", "cost": "price", "pricing": "price",
    "computer": "laptop", "pc": "laptop", "macbook": "mac", "macos": "mac",
    "powerbi": "power_bi", "certificate": "certification",
}


def content_tokens(text: str) -> list[str]:
    """Stemmed, stop-word-free, canonicalised tokens."""
    out = []
    for token in normalize(text).split():
        if token in STOPWORDS or len(token) < 2:
            continue
        stemmed = stem(token)
        out.append(CANONICAL.get(stemmed, CANONICAL.get(token, stemmed)))
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
