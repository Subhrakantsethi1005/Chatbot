"""Post-activation health checks.

A check passes when the version's files are intact, smoke questions are
answered (not escalated) within ``health.max_latency_ms``, and - once
enough live traffic exists - the chat error rate since activation stays
below ``health.max_error_rate``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable

from .config import HealthConfig
from .engine import AnswerEngine
from .faults import FaultInjector
from .monitoring import Metrics


@dataclass
class HealthResult:
    ok: bool
    version: str | None
    checks: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HealthChecker:
    def __init__(self, cfg: HealthConfig, *, engine_loader: Callable[[str], AnswerEngine], metrics: Metrics,
                 faults: FaultInjector, samples: int = 5) -> None:
        self.cfg = cfg
        self.engine_loader = engine_loader
        self.metrics = metrics
        self.faults = faults
        self.samples = samples

    def smoke_queries(self, engine: AnswerEngine) -> list[str]:
        """Configured smoke questions, or questions sampled evenly from the
        version itself (these test that the version *serves* correctly;
        answer quality was already proven by the quality gate)."""
        if self.cfg.smoke_queries:
            return list(self.cfg.smoke_queries)
        questions = [e.question for e in sorted(engine.entries.values(), key=lambda e: (e.source, e.row))]
        if len(questions) <= self.samples:
            return questions
        step = len(questions) / self.samples
        return [questions[int(i * step)] for i in range(self.samples)]

    def check(self, version: str, *, since: datetime | None = None) -> HealthResult:
        checks: dict[str, Any] = {}
        try:
            self.faults.check("health_check")
            engine = self.engine_loader(version)          # verifies checksums
            checks["integrity"] = "ok"
            failures = []
            slowest = 0.0
            queries = self.smoke_queries(engine)
            if not queries:
                return HealthResult(False, version, checks, "version has no entries to serve")
            for query in queries:
                started = time.perf_counter()
                answer = engine.answer(query)
                elapsed = (time.perf_counter() - started) * 1000
                slowest = max(slowest, elapsed)
                if answer.escalated or not answer.text.strip():
                    failures.append(f"'{query}' was not answered ({answer.reason})")
                if elapsed > self.cfg.max_latency_ms:
                    failures.append(f"'{query}' took {elapsed:.0f} ms (limit {self.cfg.max_latency_ms:g} ms)")
            checks["smoke_queries"] = len(queries)
            checks["slowest_ms"] = round(slowest, 2)
            if failures:
                return HealthResult(False, version, checks, "; ".join(failures[:5]))
            if since is not None:
                live = self.metrics.summary(window_minutes=None, since=since)["chat"]
                checks["live_requests"] = live["requests"]
                checks["live_error_rate"] = live["error_rate"]
                if live["requests"] >= self.cfg.min_samples and live["error_rate"] > self.cfg.max_error_rate:
                    return HealthResult(False, version, checks,
                                        f"live error rate {live['error_rate']:.0%} exceeds {self.cfg.max_error_rate:.0%}")
            return HealthResult(True, version, checks)
        except Exception as exc:  # any failure to serve is a failed health check
            return HealthResult(False, version, checks, f"{type(exc).__name__}: {exc}")
