"""Quality tests run against every candidate knowledge base before it may be
activated.

The evaluation set (``config/eval_set.jsonl``) holds real questions with the
facts a correct answer must contain, plus out-of-scope questions that must be
escalated instead of answered::

    {"id": "emi", "question": "Can I pay in EMIs?", "expected_keywords": ["no"],
     "expected_questions": ["Do we have an EMI option?"]}
    {"id": "oos-1", "question": "What's the weather in Paris?", "should_escalate": true}

Metrics:

* accuracy  - share of items handled correctly (right facts for in-scope
              questions, escalation for out-of-scope ones)
* grounding - share of answered items whose answer is supported by the
              retrieved knowledge-base context and cites a source

The gate rejects a candidate that falls below the absolute minimums, scores
lower than the active version on either metric (beyond ``max_regression``),
contains leftover personal data or injection text, or matches a version that
was already rolled back.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import QualityConfig
from .documents import KBEntry
from .engine import AnswerEngine
from .errors import ConfigError
from .security import InjectionDetector, PIIMasker
from .textutil import normalize


@dataclass
class EvalItem:
    id: str
    question: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_questions: list[str] = field(default_factory=list)
    should_escalate: bool = False


def load_eval_set(path: Path) -> list[EvalItem]:
    if not path.is_file():
        raise ConfigError(f"evaluation set not found: {path}")
    items: list[EvalItem] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                item = EvalItem(
                    id=str(data.get("id") or f"item-{number}"),
                    question=data["question"],
                    expected_keywords=list(data.get("expected_keywords", [])),
                    expected_questions=list(data.get("expected_questions", [])),
                    should_escalate=bool(data.get("should_escalate", False)),
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise ConfigError(f"{path}:{number}: invalid evaluation item ({exc})") from exc
            if not item.should_escalate and not (item.expected_keywords or item.expected_questions):
                raise ConfigError(f"{path}:{number}: in-scope items need expected_keywords or expected_questions")
            items.append(item)
    if not items:
        raise ConfigError(f"evaluation set {path} is empty")
    return items


@dataclass
class ItemResult:
    id: str
    correct: bool
    escalated: bool
    grounded: bool | None
    confidence: float
    matched_question: str | None
    detail: str = ""


@dataclass
class EvalResult:
    accuracy: float
    grounding: float
    total: int
    answered: int
    mean_confidence: float
    max_latency_ms: float
    items: list[ItemResult] = field(default_factory=list)

    def failures(self) -> list[ItemResult]:
        return [i for i in self.items if not i.correct or i.grounded is False]

    def to_dict(self, *, with_items: bool = False) -> dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if k != "items"}
        data["failed_items"] = [asdict(i) for i in self.failures()][:20]
        if with_items:
            data["items"] = [asdict(i) for i in self.items]
        return data


def _contains(text: str, keyword: str) -> bool:
    return f" {normalize(keyword)} " in f" {normalize(text)} "


def evaluate(engine: AnswerEngine, items: list[EvalItem], *, grounding_threshold: float = 0.8) -> EvalResult:
    results: list[ItemResult] = []
    confidences: list[float] = []
    max_latency = 0.0
    for item in items:
        started = time.perf_counter()
        answer = engine.answer(item.question)
        max_latency = max(max_latency, (time.perf_counter() - started) * 1000)
        confidences.append(answer.confidence)
        if item.should_escalate:
            results.append(ItemResult(item.id, answer.escalated, answer.escalated, None, answer.confidence,
                                      answer.matched_question,
                                      "" if answer.escalated else f"answered out-of-scope question with '{answer.matched_question}'"))
            continue
        problems: list[str] = []
        if answer.escalated:
            problems.append(f"escalated ({answer.reason}, confidence {answer.confidence:.2f})")
        else:
            missing = [k for k in item.expected_keywords if not _contains(answer.text, k)]
            if missing:
                problems.append(f"answer is missing {missing}")
            if item.expected_questions and normalize(answer.matched_question or "") not in {
                    normalize(q) for q in item.expected_questions}:
                problems.append(f"matched '{answer.matched_question}'")
        # grounding is measured separately from correctness
        grounded = None if answer.escalated else (answer.grounding >= grounding_threshold and bool(answer.sources))
        detail = problems + ([f"answer not grounded (support {answer.grounding:.2f})"] if grounded is False else [])
        results.append(ItemResult(item.id, not problems, answer.escalated, grounded, answer.confidence,
                                  answer.matched_question, "; ".join(detail)))
    answered = [r for r in results if r.grounded is not None]
    total = len(results)
    return EvalResult(
        accuracy=round(sum(r.correct for r in results) / total, 4) if total else 0.0,
        grounding=round(sum(bool(r.grounded) for r in answered) / len(answered), 4) if answered else 0.0,
        total=total,
        answered=len(answered),
        mean_confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        max_latency_ms=round(max_latency, 2),
        items=results,
    )


@dataclass
class GateDecision:
    approved: bool
    reasons: list[str]
    candidate: dict[str, Any]
    baseline: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QualityGate:
    def __init__(self, cfg: QualityConfig, masker: PIIMasker, detector: InjectionDetector) -> None:
        self.cfg = cfg
        self.masker = masker
        self.detector = detector

    def check(self, candidate: EvalResult, baseline: EvalResult | None, entries: list[KBEntry], *,
              rolled_back_match: str | None = None) -> GateDecision:
        reasons: list[str] = []
        if len(entries) < self.cfg.min_entries:
            reasons.append(f"only {len(entries)} entries (minimum {self.cfg.min_entries})")
        if candidate.accuracy < self.cfg.min_accuracy:
            reasons.append(f"accuracy {candidate.accuracy:.1%} is below the minimum {self.cfg.min_accuracy:.1%}")
        if candidate.grounding < self.cfg.min_grounding:
            reasons.append(f"grounding {candidate.grounding:.1%} is below the minimum {self.cfg.min_grounding:.1%}")
        if baseline is not None:
            if candidate.accuracy + self.cfg.max_regression < baseline.accuracy:
                reasons.append(f"accuracy would drop from {baseline.accuracy:.1%} to {candidate.accuracy:.1%}")
            if candidate.grounding + self.cfg.max_regression < baseline.grounding:
                reasons.append(f"grounding would drop from {baseline.grounding:.1%} to {candidate.grounding:.1%}")
        leaks = [e.id for e in entries if self.masker.mask(e.question + "\n" + e.answer).total]
        if leaks:
            reasons.append(f"{len(leaks)} entries still contain sensitive data")
        injected = [e.id for e in entries if self.detector.scan(e.question + "\n" + e.answer).blocked]
        if injected:
            reasons.append(f"{len(injected)} entries contain prompt-injection text")
        if rolled_back_match:
            reasons.append(f"content is identical to {rolled_back_match}, which was rolled back after failing")
        return GateDecision(not reasons, reasons, candidate.to_dict(), baseline.to_dict() if baseline else None)
