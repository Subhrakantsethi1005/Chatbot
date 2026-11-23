"""Prompt-injection detection and sensitive-data masking.

The detector is applied twice:

* to every ingested document row - a poisoned document is quarantined before
  it can reach the knowledge base;
* to every user question - injection attempts are refused and logged.

The masker removes personal data and secrets from documents before they are
indexed, from answers before they are shown, and from anything written to
logs or the escalation queue.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .textutil import INVISIBLE_RE, clean_text

# --------------------------------------------------------------------------- injection

_GAP = r"[^.\n]{0,40}?"

_INJECTION_RULES: list[tuple[str, float, re.Pattern[str]]] = [
    ("ignore_instructions", 0.85, re.compile(
        r"\b(ignore|disregard|forget|override|bypass|skip)\b" + _GAP +
        r"\b(previous|prior|above|earlier|preceding|all|any|your|these|those|the|my)\b" + _GAP +
        r"\b(instructions?|prompts?|rules|directions|guidelines|guardrails|polic(?:y|ies)|context|commands?)\b")),
    ("reveal_system_prompt", 0.85, re.compile(
        r"\b(reveal|show|print|repeat|display|output|leak|dump|tell me|what (?:is|are))\b" + _GAP +
        r"\b(system|hidden|secret|initial|original|developer|internal)\s+(prompts?|instructions?|messages?|rules)\b")),
    ("role_override", 0.6, re.compile(
        r"\byou are (?:now|no longer)\b|\bfrom now on,? you\b|\bpretend (?:to be|you are)\b|"
        r"\bact as (?:an? )?(?:unfiltered|unrestricted|evil|jailbroken|different)\b|\bnew persona\b")),
    ("jailbreak", 0.75, re.compile(
        r"\b(jailbreak|jailbroken|dan mode|developer mode|do anything now|god mode)\b|"
        r"\bwithout (?:any )?(?:restrictions|filters|limitations|guardrails)\b")),
    ("chat_template_tokens", 0.8, re.compile(
        r"<\|?\s*(?:im_start|im_end|system|endoftext|assistant)\s*\|?>|\[/?(?:inst|sys)\]|<</?sys>>")),
    ("fake_role_header", 0.55, re.compile(r"(?:^|\n)\s*(?:#+\s*)?(?:system|assistant)\s*(?:prompt)?\s*:", re.MULTILINE)),
    ("new_instructions", 0.6, re.compile(r"\b(new|updated|real|actual|secret|hidden)\s+instructions?\s*:")),
    ("hide_from_user", 0.6, re.compile(r"\bdo not (?:tell|inform|mention|reveal|show)\b" + _GAP + r"\b(user|customer|student)s?\b")),
    ("forced_response", 0.45, re.compile(
        r"\b(always|only)\s+(?:say|respond|answer|reply|tell)\b|\b(?:respond|answer|reply)\b" + _GAP +
        r"\bregardless\b")),
    ("exfiltration", 0.6, re.compile(
        r"\b(send|email|post|upload|forward|share|exfiltrate)\b" + _GAP +
        r"\b(conversation|chat history|user data|api keys?|passwords?|credentials|secrets?|tokens?)\b")),
    ("hidden_html_instruction", 0.6, re.compile(r"<!--[\s\S]{0,400}?(instruction|ignore|system|assistant)[\s\S]{0,400}?-->")),
    ("encoded_payload", 0.3, re.compile(r"[A-Za-z0-9+/]{160,}={0,2}")),
]

# Phrases checked after removing every non-letter (defeats "i.g.n.o.r.e" or
# "ign ore prev ious instruc tions" style obfuscation).
_COLLAPSED_PHRASES = [
    ("ignorepreviousinstructions", "ignore_instructions_obfuscated", 0.85),
    ("ignoreallpreviousinstructions", "ignore_instructions_obfuscated", 0.85),
    ("ignoreallinstructions", "ignore_instructions_obfuscated", 0.85),
    ("ignoretheaboveinstructions", "ignore_instructions_obfuscated", 0.85),
    ("disregardpreviousinstructions", "ignore_instructions_obfuscated", 0.85),
    ("revealyoursystemprompt", "reveal_system_prompt_obfuscated", 0.85),
    ("showmeyoursystemprompt", "reveal_system_prompt_obfuscated", 0.85),
    ("developermode", "jailbreak_obfuscated", 0.6),
    ("jailbreak", "jailbreak_obfuscated", 0.6),
]

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})


@dataclass
class InjectionResult:
    score: float
    verdict: str                       # "clean" | "suspicious" | "block"
    matches: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.verdict == "block"


class InjectionDetector:
    def __init__(self, block_threshold: float = 0.7, suspicious_threshold: float = 0.4) -> None:
        self.block_threshold = block_threshold
        self.suspicious_threshold = suspicious_threshold

    def scan(self, text: str) -> InjectionResult:
        if not text or not text.strip():
            return InjectionResult(0.0, "clean")
        hidden = len(INVISIBLE_RE.findall(unicodedata.normalize("NFKC", text)))
        cleaned = clean_text(text).lower()
        variants = {cleaned, cleaned.translate(_LEET)}
        matched: dict[str, float] = {}
        for name, weight, pattern in _INJECTION_RULES:
            if any(pattern.search(v) for v in variants):
                matched[name] = weight
        collapsed = re.sub(r"[^a-z]", "", cleaned.translate(_LEET))
        for phrase, name, weight in _COLLAPSED_PHRASES:
            if phrase in collapsed and name.replace("_obfuscated", "") not in matched:
                matched[name] = max(weight, matched.get(name, 0.0))
        if hidden:
            # invisible characters are a strong signal only alongside something else
            matched["hidden_characters"] = 0.35 if matched else (0.4 if hidden >= 3 else 0.1)
        remaining = 1.0
        for weight in matched.values():
            remaining *= 1.0 - weight
        score = round(1.0 - remaining, 3)
        if score >= self.block_threshold:
            verdict = "block"
        elif score >= self.suspicious_threshold:
            verdict = "suspicious"
        else:
            verdict = "clean"
        return InjectionResult(score, verdict, sorted(matched))


# --------------------------------------------------------------------------- masking

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
_URL_SECRET_PARAM_RE = re.compile(
    r"([?&](?:token|access_token|key|api_key|apikey|password|passwd|secret|sig|signature|auth)=)[^&#\s]+",
    re.IGNORECASE)


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _mask_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return "[CARD]"
    return match.group(0)


_IPV4 = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
# Number patterns must not start or end inside a longer group of digits
# (e.g. the last 12 digits of a 16-digit order number are not an Aadhaar).
_NOT_AFTER_DIGITS = r"(?<![\d-])(?<!\d )(?<!\d-)"
_NOT_BEFORE_DIGITS = r"(?![\d-])(?! \d)(?!-\d)"

# (label, pattern, replacement) - applied in order outside URLs
_MASK_RULES: list[tuple[str, re.Pattern[str], object]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), "[REDACTED_TOKEN]"),
    ("api_key", re.compile(
        r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
        r"|AIza[0-9A-Za-z_-]{35}|xox[abprs]-[A-Za-z0-9-]{10,}|kb_[A-Za-z0-9_-]{30,})"), "[REDACTED_API_KEY]"),
    ("bearer", re.compile(r"\b(bearer)\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE), r"\1 [REDACTED_TOKEN]"),
    ("password", re.compile(r"\b(password|passwd|pwd|passcode|pin)\b(\s*[:=]\s*)\S+", re.IGNORECASE), r"\1\2[REDACTED_PASSWORD]"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"), None),
    ("card", re.compile(_NOT_AFTER_DIGITS + r"(?:\d[ -]?){12,18}\d" + _NOT_BEFORE_DIGITS), _mask_card),
    ("aadhaar", re.compile(_NOT_AFTER_DIGITS + r"[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}" + _NOT_BEFORE_DIGITS), "[AADHAAR]"),
    ("ssn", re.compile(_NOT_AFTER_DIGITS + r"\d{3}-\d{2}-\d{4}" + _NOT_BEFORE_DIGITS), "[SSN]"),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    ("phone", re.compile(
        r"(?<![\w+])(?:(?:\+|00)91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"
        r"|(?<![\w+])\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,4}(?!\d)"
        r"|(?<!\d)\(\d{3}\)\s?\d{3}[\s.-]\d{4}(?!\d)"
        r"|(?<![\d-])\d{3}[.-]\d{3}[.-]\d{4}(?![\d-])"), "[PHONE]"),
    ("ip_address", re.compile(rf"(?<![\d.]){_IPV4}\.{_IPV4}\.{_IPV4}\.{_IPV4}(?![\d.])"), "[IP]"),
]


@dataclass
class MaskResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class PIIMasker:
    """Regex-based masking of personal data and secrets. Numbers inside URLs
    (for example Discord/LinkedIn message IDs) are left alone; secret-looking
    query parameters in URLs are still redacted."""

    def __init__(self, allowlist: tuple[str, ...] = ()) -> None:
        self.allowlist = {a.lower() for a in allowlist}

    def _allowed_email(self, email: str) -> bool:
        email = email.lower()
        domain = email.split("@", 1)[-1]
        return email in self.allowlist or f"@{domain}" in self.allowlist

    def mask(self, text: str) -> MaskResult:
        if not text:
            return MaskResult(text or "")
        counts: dict[str, int] = {}
        pieces: list[str] = []
        last = 0
        for url in _URL_RE.finditer(text):
            pieces.append(self._mask_plain(text[last:url.start()], counts))
            masked_url, n = _URL_SECRET_PARAM_RE.subn(r"\1[REDACTED]", url.group(0))
            if n:
                counts["url_secret"] = counts.get("url_secret", 0) + n
            pieces.append(masked_url)
            last = url.end()
        pieces.append(self._mask_plain(text[last:], counts))
        return MaskResult("".join(pieces), counts)

    def _mask_plain(self, text: str, counts: dict[str, int]) -> str:
        if not text:
            return text
        for label, pattern, replacement in _MASK_RULES:
            hits = 0

            def substitute(match: re.Match[str], label=label, replacement=replacement) -> str:
                nonlocal hits
                original = match.group(0)
                if label == "email":
                    if self._allowed_email(original):
                        return original
                    out = "[EMAIL]"
                elif callable(replacement):
                    out = replacement(match)
                else:
                    out = match.expand(replacement)  # type: ignore[arg-type]
                if out != original:
                    hits += 1
                return out

            text = pattern.sub(substitute, text)
            if hits:
                counts[label] = counts.get(label, 0) + hits
        return text


def mask_text(text: str, masker: PIIMasker | None = None) -> str:
    return (masker or PIIMasker()).mask(text).text
