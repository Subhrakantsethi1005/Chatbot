"""The runtime chat service used by the Streamlit app.

Per question it: checks access, refuses prompt-injection attempts, answers
from the *active* knowledge-base version, masks personal data in the reply,
escalates low-confidence or explicit "talk to a human" requests to the
support queue, and records latency / confidence / escalation / failure
metrics.

Outages degrade gracefully: if the active version cannot be read, the last
version loaded in memory keeps serving (marked ``degraded``); if nothing is
loaded at all, the question is escalated with an apology instead of failing.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .access import AccessController
from .clock import Clock, SystemClock
from .config import PipelineConfig
from .engine import BLOCKED_MESSAGE, OUTAGE_MESSAGE, AnswerEngine, CircuitBreaker, Generator
from .errors import StoreError, TransientError
from .faults import FaultInjector
from .monitoring import AlertManager, JsonlLog, Metrics
from .security import InjectionDetector, PIIMasker
from .store import VersionStore

_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+\]")


@dataclass
class ChatResponse:
    answer: str
    confidence: float
    escalated: bool
    blocked: bool
    reason: str
    version: str | None
    latency_ms: float
    matched_question: str | None = None
    sources: list[str] = field(default_factory=list)
    degraded: bool = False
    ticket_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChatService:
    def __init__(self, cfg: PipelineConfig, *, clock: Clock | None = None, faults: FaultInjector | None = None,
                 generator: Generator | None = None, access: AccessController | None = None) -> None:
        self.cfg = cfg
        self.clock = clock or SystemClock(cfg.tz)
        self.faults = faults if faults is not None else FaultInjector.from_env()
        self.generator = generator
        logs = cfg.paths.data_dir / "logs"
        self.store = VersionStore(cfg.paths.data_dir / "versions", faults=self.faults)
        self.events = JsonlLog(logs / "events.jsonl", self.clock)
        self.metrics = Metrics(self.events, self.clock)
        self.alerts = AlertManager(logs / "alerts.jsonl", self.clock, cfg.monitoring, self.metrics,
                                   min_confidence=cfg.retrieval.min_confidence)
        self.audit = JsonlLog(logs / "audit.jsonl", self.clock)
        self.escalations = JsonlLog(logs / "escalations.jsonl", self.clock)
        self.access = access or AccessController(cfg.paths.data_dir / "access.json", clock=self.clock,
                                                 cfg=cfg.security, audit=self.audit, metrics=self.metrics,
                                                 alerts=self.alerts)
        self.masker = PIIMasker(cfg.security.pii_allowlist)
        self.detector = InjectionDetector(cfg.security.injection_block_threshold,
                                          cfg.security.injection_suspicious_threshold)
        self.breaker = CircuitBreaker()
        self._engine: AnswerEngine | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ engine
    def _current_engine(self) -> tuple[AnswerEngine | None, bool, str | None]:
        """(engine, degraded, problem)."""
        with self._lock:
            try:
                version = self.store.active_version()
            except (StoreError, TransientError, OSError) as exc:
                return self._engine, self._engine is not None, f"active pointer unreadable: {exc}"
            if version is None:
                return self._engine, self._engine is not None, "no knowledge-base version is active yet"
            if self._engine is not None and self._engine.version == version:
                return self._engine, False, None
            try:
                engine = AnswerEngine(version, self.store.load_entries(version), self.store.load_index(version),
                                      self.cfg.retrieval, generator=self.generator, breaker=self.breaker,
                                      grounding_threshold=self.cfg.quality.grounding_threshold, faults=self.faults)
            except (StoreError, TransientError, OSError) as exc:
                self.alerts.raise_alert("kb_load_failed", "critical", f"cannot load {version}: {exc}", version=version)
                return self._engine, self._engine is not None, f"cannot load {version}: {exc}"
            self._engine = engine
            return engine, False, None

    def active_version(self) -> str | None:
        engine, _, _ = self._current_engine()
        return engine.version if engine else None

    # ------------------------------------------------------------------ chat
    def _ticket(self, question: str, reason: str, version: str | None, confidence: float) -> str:
        ticket = "esc-" + secrets.token_hex(4)
        self.escalations.append("escalation", ticket=ticket, question=self.masker.mask(question).text,
                                reason=reason, version=version, confidence=round(confidence, 4))
        return ticket

    def ask(self, question: str, *, token: str | None = None, client_id: str = "anonymous") -> ChatResponse:
        """Answer a question. Raises AccessDenied if the caller may not chat."""
        started = time.perf_counter()
        self.access.authorize(token, "chat:ask", client_id=client_id)
        question = (question or "").strip()

        def finish(response: ChatResponse, *, error: bool = False, fallback: bool = False, pii: int = 0) -> ChatResponse:
            response.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self.metrics.record_chat(latency_ms=response.latency_ms, confidence=response.confidence,
                                     escalated=response.escalated, blocked=response.blocked, error=error,
                                     version=response.version, reason=response.reason,
                                     degraded=response.degraded, fallback=fallback, pii_masked=pii)
            return response

        if not question:
            return ChatResponse("Please type a question.", 0.0, False, True, "empty", None, 0.0)
        if len(question) > self.cfg.security.max_query_chars:
            return finish(ChatResponse(f"Please keep questions under {self.cfg.security.max_query_chars} characters.",
                                       0.0, False, True, "too_long", None, 0.0))
        verdict = self.detector.scan(question)
        if verdict.blocked:
            self.audit.append("security", event="prompt_injection_blocked", client=client_id,
                              score=verdict.score, matches=verdict.matches)
            return finish(ChatResponse(BLOCKED_MESSAGE, 0.0, False, True, "prompt_injection", None, 0.0))

        # Personal data in the question is masked before it is searched,
        # logged or queued for staff; placeholders are not search terms.
        masked_question = self.masker.mask(question)
        search_text = _PLACEHOLDER_RE.sub(" ", masked_question.text)

        engine, degraded, problem = self._current_engine()
        if engine is None:
            ticket = self._ticket(question, "kb_unavailable", None, 0.0)
            return finish(ChatResponse(OUTAGE_MESSAGE, 0.0, True, False, "kb_unavailable", None, 0.0,
                                       degraded=True, ticket_id=ticket), error=True, pii=masked_question.total)
        try:
            answer = engine.answer(search_text)
        except Exception:
            ticket = self._ticket(question, "engine_error", engine.version, 0.0)
            return finish(ChatResponse(OUTAGE_MESSAGE, 0.0, True, False, "engine_error", engine.version, 0.0,
                                       degraded=True, ticket_id=ticket), error=True, pii=masked_question.total)

        masked = self.masker.mask(answer.text)
        ticket = self._ticket(question, answer.reason, engine.version, answer.confidence) if answer.escalated else None
        response = ChatResponse(masked.text, answer.confidence, answer.escalated, False, answer.reason,
                                engine.version, 0.0, answer.matched_question, answer.sources,
                                degraded=degraded, ticket_id=ticket)
        return finish(response, fallback=answer.fallback, pii=masked.total + masked_question.total)
