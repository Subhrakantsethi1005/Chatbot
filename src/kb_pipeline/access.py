"""Token-based access control with roles, lockout and an audit trail.

Roles and what they may do:

=========  ===============================================================
chat       ask questions (anonymous users get this role when
           ``security.allow_anonymous_chat`` is true)
viewer     read status, versions, metrics, alerts, quarantine, escalations
operator   viewer + run updates, run the scheduler, activate approved updates
admin      operator + roll back, manage quarantine, create/revoke tokens
=========  ===============================================================

Tokens are random 256-bit strings shown once at creation; only their SHA-256
hash is stored (``data_dir/access.json``). Repeated failures from one client
lock it out for ``security.lockout_minutes``. Every decision is written to
``logs/audit.jsonl``.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .clock import Clock
from .config import SecurityConfig
from .errors import AccessDenied, AuthLockedOut
from .fsutil import atomic_write_json, read_json
from .monitoring import JsonlLog, Metrics

_VIEW = {"kb:read", "metrics:read", "quarantine:read", "escalations:read", "chat:ask"}
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "chat": frozenset({"chat:ask"}),
    "viewer": frozenset(_VIEW),
    "operator": frozenset(_VIEW | {"kb:update", "kb:activate"}),
    "admin": frozenset(_VIEW | {"kb:update", "kb:activate", "kb:rollback", "quarantine:manage", "access:manage"}),
}
_NAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


@dataclass(frozen=True)
class Principal:
    name: str
    role: str

    @property
    def permissions(self) -> frozenset[str]:
        return ROLE_PERMISSIONS.get(self.role, frozenset())


ANONYMOUS = Principal("anonymous", "chat")
SYSTEM = Principal("system", "admin")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AccessController:
    def __init__(self, path: Path, *, clock: Clock, cfg: SecurityConfig, audit: JsonlLog,
                 metrics: Metrics | None = None, alerts: Any = None) -> None:
        self.path = path
        self.failures_path = path.with_name("auth_failures.json")
        self.clock = clock
        self.cfg = cfg
        self.audit = audit
        self.metrics = metrics
        self.alerts = alerts

    # ------------------------------------------------------------------ users
    def _users(self) -> list[dict[str, Any]]:
        return (read_json(self.path, {"users": []}) or {"users": []}).get("users", [])

    def _save_users(self, users: list[dict[str, Any]]) -> None:
        atomic_write_json(self.path, {"users": users})

    def has_users(self) -> bool:
        return any(not u.get("disabled") for u in self._users())

    def list_users(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in u.items() if k != "token_sha256"} for u in self._users()]

    def _issue(self, name: str, role: str, actor: str) -> str:
        if role not in ROLE_PERMISSIONS:
            raise ValueError(f"unknown role {role!r}; choose from {', '.join(ROLE_PERMISSIONS)}")
        if not _NAME_RE.match(name):
            raise ValueError("name may contain only letters, digits and _ . @ - (max 64)")
        users = self._users()
        if any(u["name"] == name and not u.get("disabled") for u in users):
            raise ValueError(f"user {name!r} already exists; revoke it first to issue a new token")
        token = "kb_" + secrets.token_urlsafe(32)
        users = [u for u in users if u["name"] != name]
        users.append({"name": name, "role": role, "token_sha256": hash_token(token),
                      "created_at": self.clock.now().isoformat(), "created_by": actor, "disabled": False})
        self._save_users(users)
        self.audit.append("access_change", action="create_user", user=name, role=role, actor=actor)
        return token

    def bootstrap(self, name: str = "admin") -> str:
        """Create the first admin token. Only allowed while no users exist."""
        if self.has_users():
            self._audit_denied("bootstrap", None, "local", "users already exist")
            raise AccessDenied("bootstrap is only allowed before any user exists")
        return self._issue(name, "admin", actor="bootstrap")

    def create_user(self, actor: Principal, name: str, role: str) -> str:
        if "access:manage" not in actor.permissions:
            raise AccessDenied(f"role '{actor.role}' cannot manage access")
        return self._issue(name, role, actor=actor.name)

    def revoke(self, actor: Principal, name: str) -> bool:
        if "access:manage" not in actor.permissions:
            raise AccessDenied(f"role '{actor.role}' cannot manage access")
        users = self._users()
        changed = False
        for user in users:
            if user["name"] == name and not user.get("disabled"):
                user["disabled"] = True
                user["disabled_at"] = self.clock.now().isoformat()
                changed = True
        if changed:
            self._save_users(users)
            self.audit.append("access_change", action="revoke_user", user=name, actor=actor.name)
        return changed

    # ------------------------------------------------------------------ lockout
    def _failures(self) -> dict[str, list[str]]:
        return read_json(self.failures_path, {}) or {}

    def _recent_failures(self, client_id: str, now: datetime) -> list[datetime]:
        window = timedelta(minutes=max(self.cfg.failed_auth_window_minutes, self.cfg.lockout_minutes))
        stamps = []
        for stamp in self._failures().get(client_id, []):
            try:
                at = datetime.fromisoformat(stamp)
            except ValueError:
                continue
            if now - at <= window:
                stamps.append(at)
        return stamps

    def locked_until(self, client_id: str) -> datetime | None:
        now = self.clock.now()
        stamps = self._recent_failures(client_id, now)
        window = timedelta(minutes=self.cfg.failed_auth_window_minutes)
        recent = sorted(stamps)
        # locked when max_failed_auth failures happened within the failure window
        for i in range(len(recent) - self.cfg.max_failed_auth + 1):
            first, last = recent[i], recent[i + self.cfg.max_failed_auth - 1]
            if last - first <= window:
                until = last + timedelta(minutes=self.cfg.lockout_minutes)
                if until > now:
                    return until
        return None

    def _record_failure(self, client_id: str) -> None:
        now = self.clock.now()
        data = self._failures()
        keep = [s.isoformat() for s in self._recent_failures(client_id, now)]
        keep.append(now.isoformat())
        data[client_id] = keep[-50:]
        atomic_write_json(self.failures_path, data)
        if self.locked_until(client_id):
            self.audit.append("auth_lockout", client=client_id, failures=len(keep))
            if self.metrics:
                self.metrics.record("auth_lockout", client=client_id)
            if self.alerts:
                self.alerts.raise_alert("auth_lockout", "critical",
                                        f"client {client_id} locked out after {self.cfg.max_failed_auth} failed logins",
                                        client=client_id)

    def _clear_failures(self, client_id: str) -> None:
        data = self._failures()
        if client_id in data:
            data.pop(client_id)
            atomic_write_json(self.failures_path, data)

    # ------------------------------------------------------------------ checks
    def _audit_denied(self, permission: str, principal: Principal | None, client_id: str, reason: str) -> None:
        self.audit.append("access", allowed=False, permission=permission, client=client_id,
                          user=principal.name if principal else None, reason=reason)
        if self.metrics:
            self.metrics.record("auth_denied", permission=permission, client=client_id, reason=reason)

    def authenticate(self, token: str | None, client_id: str = "local") -> Principal:
        until = self.locked_until(client_id)
        if until:
            self._audit_denied("authenticate", None, client_id, "locked_out")
            raise AuthLockedOut(f"too many failed attempts; try again after {until:%H:%M}")
        if not token:
            self._audit_denied("authenticate", None, client_id, "missing_token")
            raise AccessDenied("authentication required: pass --token or set KB_TOKEN")
        digest = hash_token(token.strip())
        match = None
        for user in self._users():
            if hmac.compare_digest(user.get("token_sha256", ""), digest):
                match = user
        if match is None or match.get("disabled"):
            self._audit_denied("authenticate", None, client_id, "invalid_token" if match is None else "revoked_token")
            self._record_failure(client_id)
            raise AccessDenied("invalid or revoked token")
        self._clear_failures(client_id)
        return Principal(match["name"], match["role"])

    def authorize(self, token: str | None, permission: str, *, client_id: str = "local",
                  action: str | None = None) -> Principal:
        if permission == "chat:ask" and not token and self.cfg.allow_anonymous_chat:
            if self.locked_until(client_id):
                self._audit_denied(permission, None, client_id, "locked_out")
                raise AuthLockedOut("too many failed attempts from this client")
            return ANONYMOUS
        principal = self.authenticate(token, client_id)
        if permission not in principal.permissions:
            self._audit_denied(permission, principal, client_id, "forbidden")
            raise AccessDenied(f"user '{principal.name}' (role '{principal.role}') may not perform '{permission}'")
        self.audit.append("access", allowed=True, permission=permission, client=client_id,
                          user=principal.name, action=action or permission)
        return principal
