"""Event log, metrics summaries and alerting.

Every chat answer, update, activation, rollback, health check and denied
access is appended as one JSON line to ``logs/events.jsonl``. No raw user
questions are stored there (only numbers and reasons), so the log is safe to
share. ``summary()`` turns a time window of events into latency percentiles,
failure / escalation rates and confidence statistics; ``AlertManager``
compares those with the thresholds in ``[monitoring]`` and records alerts
in ``logs/alerts.jsonl``.
"""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

from .clock import Clock
from .config import MonitoringConfig

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path.resolve()), threading.Lock())


class JsonlLog:
    def __init__(self, path: Path, clock: Clock, *, max_bytes: int = 20_000_000) -> None:
        self.path = path
        self.clock = clock
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, **fields: Any) -> dict[str, Any]:
        record = {"ts": self.clock.now().isoformat(), "kind": kind, **fields}
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with _lock_for(self.path):
            try:
                if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                    rotated = self.path.with_suffix(self.path.suffix + ".1")
                    rotated.unlink(missing_ok=True)
                    self.path.rename(rotated)
            except OSError:
                pass
            with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
        return record

    def read(self, *, since: datetime | None = None, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        wanted = set(kinds) if kinds else None
        records: list[dict[str, Any]] = []
        files = [self.path.with_suffix(self.path.suffix + ".1"), self.path]
        for file in files:
            if not file.is_file():
                continue
            with open(file, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if wanted and record.get("kind") not in wanted:
                        continue
                    if since is not None:
                        try:
                            if datetime.fromisoformat(record["ts"]) < since:
                                continue
                        except (KeyError, ValueError):
                            continue
                    records.append(record)
        return records

    def tail(self, limit: int = 20, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        return self.read(kinds=kinds)[-limit:]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return round(ordered[rank], 2)


class Metrics:
    def __init__(self, log: JsonlLog, clock: Clock) -> None:
        self.log = log
        self.clock = clock

    def record(self, kind: str, **fields: Any) -> dict[str, Any]:
        return self.log.append(kind, **fields)

    def record_chat(self, *, latency_ms: float, confidence: float, escalated: bool, blocked: bool,
                    error: bool, version: str | None, reason: str, degraded: bool = False,
                    fallback: bool = False, pii_masked: int = 0) -> None:
        self.record("chat", latency_ms=round(latency_ms, 2), confidence=round(confidence, 4),
                    escalated=escalated, blocked=blocked, error=error, version=version,
                    reason=reason, degraded=degraded, fallback=fallback, pii_masked=pii_masked)

    def summary(self, *, window_minutes: float | None = 60, since: datetime | None = None,
                min_confidence: float = 0.3) -> dict[str, Any]:
        if since is None and window_minutes:
            since = self.clock.now() - timedelta(minutes=window_minutes)
        events = self.log.read(since=since)
        chats = [e for e in events if e.get("kind") == "chat"]
        answered = [e for e in chats if not e.get("blocked")]
        latencies = [float(e.get("latency_ms", 0)) for e in chats]
        confidences = [float(e.get("confidence", 0)) for e in answered if not e.get("error")]
        count = len(chats)

        def rate(n: int) -> float:
            return round(n / count, 4) if count else 0.0

        def of(kind: str, **match: Any) -> list[dict[str, Any]]:
            return [e for e in events if e.get("kind") == kind and all(e.get(k) == v for k, v in match.items())]

        updates = of("update")
        statuses: dict[str, int] = {}
        for u in updates:
            statuses[u.get("status", "unknown")] = statuses.get(u.get("status", "unknown"), 0) + 1
        errors = sum(1 for e in chats if e.get("error"))
        escalations = sum(1 for e in chats if e.get("escalated"))
        return {
            "since": since.isoformat() if since else None,
            "chat": {
                "requests": count,
                "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
                               "p99": percentile(latencies, 99), "max": max(latencies) if latencies else None},
                "errors": errors,
                "error_rate": rate(errors),
                "escalations": escalations,
                "escalation_rate": rate(escalations),
                "blocked": sum(1 for e in chats if e.get("blocked")),
                "degraded": sum(1 for e in chats if e.get("degraded")),
                "generator_fallbacks": sum(1 for e in chats if e.get("fallback")),
                "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
                "low_confidence_rate": round(sum(1 for c in confidences if c < min_confidence) / len(confidences), 4)
                if confidences else 0.0,
                "pii_masked": sum(int(e.get("pii_masked", 0)) for e in chats),
            },
            "updates": {
                "total": len(updates),
                "by_status": statuses,
                "retries_scheduled": len(of("update_retry_scheduled")),
                "failed_permanently": len(of("update_failed_permanently")),
                "quarantined_files": len(of("quarantine")),
                "duplicate_documents": sum(int(u.get("duplicate_documents", 0)) for u in updates),
                "duplicate_entries": sum(int(u.get("duplicate_entries", 0)) for u in updates),
            },
            "activations": len(of("activation")),
            "rollbacks": {"total": len(of("rollback")), "automatic": len(of("rollback", automatic=True))},
            "health_checks": {"passed": len(of("health_check", ok=True)), "failed": len(of("health_check", ok=False))},
            "security": {
                "auth_denied": len(of("auth_denied")),
                "lockouts": len(of("auth_lockout")),
                "injection_blocked": sum(1 for e in chats if e.get("reason") == "prompt_injection"),
            },
        }


class AlertManager:
    """Raises de-duplicated alerts. ``notifiers`` are optional callables
    (for example a function posting to Slack or e-mail) called per alert."""

    def __init__(self, path: Path, clock: Clock, cfg: MonitoringConfig, metrics: Metrics,
                 notifiers: list[Callable[[dict[str, Any]], None]] | None = None,
                 min_confidence: float = 0.3) -> None:
        self.log = JsonlLog(path, clock)
        self.clock = clock
        self.cfg = cfg
        self.metrics = metrics
        self.notifiers = notifiers or []
        self.min_confidence = min_confidence

    def raise_alert(self, kind: str, severity: str, message: str, *, dedupe: bool = True, **details: Any) -> bool:
        cooldown = timedelta(minutes=self.cfg.alert_cooldown_minutes)
        now = self.clock.now()
        if dedupe and cooldown.total_seconds() > 0 and self.log.read(since=now - cooldown, kinds=[kind]):
            return False  # same alert already raised within the cool-down period
        record = self.log.append(kind, severity=severity, message=message, **details)
        self.metrics.record("alert", alert=kind, severity=severity)
        for notify in self.notifiers:
            try:
                notify(record)
            except Exception:  # a broken notifier must never break the pipeline
                pass
        return True

    def evaluate(self) -> list[str]:
        """Check the rolling window against thresholds; returns raised alert kinds."""
        s = self.metrics.summary(window_minutes=self.cfg.window_minutes, min_confidence=self.min_confidence)
        chat = s["chat"]
        raised: list[str] = []
        enough = chat["requests"] >= self.cfg.min_samples
        checks = [
            (enough and (chat["latency_ms"]["p95"] or 0) > self.cfg.alert_p95_latency_ms, "high_latency", "warning",
             f"p95 latency {chat['latency_ms']['p95']} ms exceeds {self.cfg.alert_p95_latency_ms} ms"),
            (enough and chat["error_rate"] > self.cfg.alert_error_rate, "high_error_rate", "critical",
             f"chat error rate {chat['error_rate']:.1%} exceeds {self.cfg.alert_error_rate:.1%}"),
            (enough and chat["escalation_rate"] > self.cfg.alert_escalation_rate, "high_escalation_rate", "warning",
             f"escalation rate {chat['escalation_rate']:.1%} exceeds {self.cfg.alert_escalation_rate:.1%}"),
            (enough and chat["avg_confidence"] is not None and chat["avg_confidence"] < self.cfg.alert_min_avg_confidence,
             "low_confidence", "warning",
             f"average confidence {chat['avg_confidence']} is below {self.cfg.alert_min_avg_confidence}"),
            (s["security"]["auth_denied"] >= self.cfg.alert_auth_failures, "unauthorised_access", "critical",
             f"{s['security']['auth_denied']} denied access attempts in the last {self.cfg.window_minutes:g} minutes"),
        ]
        for condition, kind, severity, message in checks:
            if condition and self.raise_alert(kind, severity, message):
                raised.append(kind)
        return raised

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.log.tail(limit)
