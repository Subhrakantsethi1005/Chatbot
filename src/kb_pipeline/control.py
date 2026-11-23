"""Access-controlled entry points for operating the pipeline. The CLI and
the monitoring page go through this class, never straight to the pipeline,
so every operation is authorised and audited."""

from __future__ import annotations

import time
from typing import Any, Callable

from .access import AccessController, Principal
from .clock import Clock, FakeClock
from .config import PipelineConfig
from .engine import Generator
from .errors import StoreError
from .faults import FaultInjector
from .monitoring import JsonlLog
from .pipeline import KnowledgeBasePipeline


class ControlPlane:
    def __init__(self, cfg: PipelineConfig, *, clock: Clock | None = None, faults: FaultInjector | None = None,
                 generator: Generator | None = None, client_id: str = "cli") -> None:
        self.cfg = cfg
        self.pipeline = KnowledgeBasePipeline(cfg, clock=clock, faults=faults, generator=generator)
        self.access = AccessController(cfg.paths.data_dir / "access.json", clock=self.pipeline.clock,
                                       cfg=cfg.security, audit=self.pipeline.audit, metrics=self.pipeline.metrics,
                                       alerts=self.pipeline.alerts)
        self.client_id = client_id

    def _auth(self, token: str | None, permission: str, action: str) -> Principal:
        return self.access.authorize(token, permission, client_id=self.client_id, action=action)

    # ------------------------------------------------------------------ users
    def bootstrap(self, name: str = "admin") -> str:
        return self.access.bootstrap(name)

    def create_user(self, token: str | None, name: str, role: str) -> str:
        return self.access.create_user(self._auth(token, "access:manage", "create_user"), name, role)

    def revoke_user(self, token: str | None, name: str) -> bool:
        return self.access.revoke(self._auth(token, "access:manage", "revoke_user"), name)

    def users(self, token: str | None) -> list[dict[str, Any]]:
        self._auth(token, "access:manage", "list_users")
        return self.access.list_users()

    def audit_log(self, token: str | None, limit: int = 50) -> list[dict[str, Any]]:
        self._auth(token, "access:manage", "read_audit")
        return self.pipeline.audit.tail(limit)

    # ------------------------------------------------------------------ operations
    def update(self, token: str | None) -> dict[str, Any]:
        principal = self._auth(token, "kb:update", "run_update")
        return self.pipeline.run_update(trigger="manual", actor=principal.name).to_dict()

    def tick(self, token: str | None) -> dict[str, Any]:
        self._auth(token, "kb:update", "tick")
        return self.pipeline.tick()

    def daemon(self, token: str | None, *, iterations: int | None = None,
               sleep: Callable[[float], None] = time.sleep,
               on_tick: Callable[[dict[str, Any]], None] | None = None) -> int:
        """Run the scheduler loop. With a FakeClock (``--now``) the clock is
        advanced by ``poll_seconds`` per iteration instead of sleeping, which
        simulates a whole night of schedules and windows in seconds."""
        self._auth(token, "kb:update", "daemon")
        clock = self.pipeline.clock
        interval = self.cfg.schedule.poll_seconds
        count = 0
        while iterations is None or count < iterations:
            result = self.pipeline.tick()
            if on_tick:
                on_tick(result)
            count += 1
            if isinstance(clock, FakeClock):
                clock.advance(seconds=interval)
            else:
                sleep(interval)
        return count

    def activate(self, token: str | None) -> dict[str, Any]:
        principal = self._auth(token, "kb:activate", "activate")
        return self.pipeline.activate_pending(actor=principal.name)

    def rollback(self, token: str | None, *, to: str | None = None, reason: str = "manual rollback") -> dict[str, Any]:
        principal = self._auth(token, "kb:rollback", "rollback")
        return self.pipeline.rollback(actor=principal.name, to=to, reason=reason)

    # ------------------------------------------------------------------ read-only
    def status(self, token: str | None) -> dict[str, Any]:
        self._auth(token, "kb:read", "status")
        return self.pipeline.status()

    def versions(self, token: str | None) -> list[dict[str, Any]]:
        self._auth(token, "kb:read", "versions")
        return self.pipeline.versions()

    def health(self, token: str | None) -> dict[str, Any]:
        self._auth(token, "kb:read", "health")
        active = self.pipeline.store.active_version()
        if not active:
            raise StoreError("no active version to check")
        return self.pipeline.health.check(active).to_dict()

    def metrics(self, token: str | None, *, window_minutes: float = 60) -> dict[str, Any]:
        self._auth(token, "metrics:read", "metrics")
        return self.pipeline.metrics.summary(window_minutes=window_minutes,
                                             min_confidence=self.cfg.retrieval.min_confidence)

    def alerts(self, token: str | None, *, limit: int = 20) -> list[dict[str, Any]]:
        self._auth(token, "metrics:read", "alerts")
        return self.pipeline.alerts.recent(limit)

    def escalations(self, token: str | None, *, limit: int = 20) -> list[dict[str, Any]]:
        self._auth(token, "escalations:read", "escalations")
        return JsonlLog(self.pipeline.logs_dir / "escalations.jsonl", self.pipeline.clock).tail(limit)

    def quarantine(self, token: str | None) -> list[dict[str, Any]]:
        self._auth(token, "quarantine:read", "quarantine")
        return self.pipeline.quarantine_list()

    def release(self, token: str | None, path: str) -> dict[str, Any]:
        principal = self._auth(token, "quarantine:manage", "release_quarantine")
        return self.pipeline.release_quarantine(path, actor=principal.name)
