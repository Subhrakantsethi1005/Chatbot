"""Immutable, versioned knowledge-base storage with activation history and
rollback.

Layout (under ``data_dir/versions``)::

    v000001/entries.jsonl   the knowledge entries (already masked)
    v000001/index.json      the search index built from them
    v000001/manifest.json   source path -> hash / entry ids (for incremental updates)
    v000001/meta.json       status, parent, checksums, quality metrics, report
    active.json             {"version": ..., "previous": ..., "activated_at": ...}
    history.jsonl           every activation / rollback

Versions are written to a temporary folder and renamed into place, and every
file's SHA-256 is recorded in meta.json and verified on load, so a partially
written or corrupted version can never be activated or served.

Status lifecycle: candidate -> approved | rejected; approved -> active |
superseded; active -> retired (replaced) | rolled_back (health failure or
manual rollback).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .documents import KBEntry
from .errors import StoreError, TransientError
from .faults import FaultInjector
from .fsutil import atomic_write_json, read_json
from .index import TfidfIndex

_VERSION_RE = re.compile(r"^v(\d{6})$")
TERMINAL_BAD = {"rejected", "rolled_back", "superseded"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VersionStore:
    def __init__(self, root: Path, *, faults: FaultInjector | None = None) -> None:
        self.root = root
        self.faults = faults or FaultInjector()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    @property
    def active_file(self) -> Path:
        return self.root / "active.json"

    @property
    def history_file(self) -> Path:
        return self.root / "history.jsonl"

    def path(self, version: str) -> Path:
        if not _VERSION_RE.match(version or ""):
            raise StoreError(f"invalid version id: {version!r}")
        return self.root / version

    def list_versions(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and _VERSION_RE.match(p.name))

    def exists(self, version: str) -> bool:
        return self.path(version).is_dir()

    def _next_id(self) -> str:
        existing = [int(_VERSION_RE.match(v).group(1)) for v in self.list_versions()]  # type: ignore[union-attr]
        return f"v{(max(existing) + 1 if existing else 1):06d}"

    # ------------------------------------------------------------------ write
    def create_version(self, entries: list[KBEntry], index: TfidfIndex, manifest: dict[str, Any],
                       meta: dict[str, Any], source_entries: list[KBEntry] | None = None) -> str:
        """``entries`` are the de-duplicated entries that get served;
        ``source_entries`` are all parsed entries per source (before
        de-duplication) and are what unchanged files carry forward."""
        self.faults.check("store_write")
        version = self._next_id()
        tmp = self.root / f".tmp-{version}-{os.getpid()}"
        try:
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            for name, rows in (("entries.jsonl", entries), ("source_entries.jsonl", source_entries or entries)):
                with open(tmp / name, "w", encoding="utf-8", newline="\n") as handle:
                    for entry in rows:
                        handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            with open(tmp / "index.json", "w", encoding="utf-8") as handle:
                json.dump(index.to_dict(), handle, ensure_ascii=False)
            with open(tmp / "manifest.json", "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
            full_meta = dict(meta)
            full_meta.update({
                "version": version,
                "entry_count": len(entries),
                "checksums": {name: _sha(tmp / name) for name in
                              ("entries.jsonl", "source_entries.jsonl", "index.json", "manifest.json")},
            })
            with open(tmp / "meta.json", "w", encoding="utf-8") as handle:
                json.dump(full_meta, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            os.replace(tmp, self.root / version)
        except OSError as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise TransientError(f"could not write version {version}: {exc}") from exc
        return version

    def update_meta(self, version: str, **fields: Any) -> dict[str, Any]:
        meta = self.load_meta(version)
        meta.update(fields)
        atomic_write_json(self.path(version) / "meta.json", meta)
        return meta

    def set_status(self, version: str, status: str, **fields: Any) -> None:
        self.update_meta(version, status=status, **fields)

    # ------------------------------------------------------------------ read
    def load_meta(self, version: str) -> dict[str, Any]:
        meta = read_json(self.path(version) / "meta.json")
        if meta is None:
            raise StoreError(f"version {version} not found")
        return meta

    def verify(self, version: str) -> None:
        """Raise StoreError if any file of the version is missing or altered."""
        meta = self.load_meta(version)
        folder = self.path(version)
        for name, expected in (meta.get("checksums") or {}).items():
            file = folder / name
            if not file.is_file():
                raise StoreError(f"version {version} is corrupted: {name} is missing")
            if _sha(file) != expected:
                raise StoreError(f"version {version} is corrupted: {name} checksum mismatch")

    def load_entries(self, version: str, *, verify: bool = True, file: str = "entries.jsonl") -> list[KBEntry]:
        self.faults.check("store_read")
        if verify:
            self.verify(version)
        try:
            with open(self.path(version) / file, encoding="utf-8") as handle:
                return [KBEntry.from_dict(json.loads(line)) for line in handle if line.strip()]
        except (OSError, ValueError, KeyError) as exc:
            raise StoreError(f"cannot read {file} of {version}: {exc}") from exc

    def load_source_entries(self, version: str) -> list[KBEntry]:
        return self.load_entries(version, file="source_entries.jsonl")

    def load_index(self, version: str, *, verify: bool = True) -> TfidfIndex:
        self.faults.check("store_read")
        if verify:
            self.verify(version)
        try:
            with open(self.path(version) / "index.json", encoding="utf-8") as handle:
                return TfidfIndex.from_dict(json.load(handle))
        except (OSError, ValueError, KeyError) as exc:
            raise StoreError(f"cannot read index of {version}: {exc}") from exc

    def load_manifest(self, version: str) -> dict[str, Any]:
        manifest = read_json(self.path(version) / "manifest.json")
        if manifest is None:
            raise StoreError(f"manifest of {version} is missing")
        return manifest

    # ------------------------------------------------------------------ activation
    def active(self) -> dict[str, Any] | None:
        try:
            return read_json(self.active_file)
        except (OSError, ValueError) as exc:
            raise StoreError(f"cannot read active pointer: {exc}") from exc

    def active_version(self) -> str | None:
        pointer = self.active()
        return pointer.get("version") if pointer else None

    def history(self) -> list[dict[str, Any]]:
        if not self.history_file.is_file():
            return []
        out = []
        with open(self.history_file, encoding="utf-8") as handle:
            for line in handle:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out

    def _append_history(self, record: dict[str, Any]) -> None:
        with open(self.history_file, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def activate(self, version: str, *, at: datetime, actor: str, reason: str) -> str | None:
        """Make ``version`` active. Returns the previously active version."""
        self.verify(version)
        previous = self.active_version()
        if previous == version:
            return previous
        atomic_write_json(self.active_file, {"version": version, "previous": previous,
                                             "activated_at": at.isoformat(), "actor": actor})
        if previous and self.exists(previous):
            self.set_status(previous, "retired", retired_at=at.isoformat())
        self.set_status(version, "active", activated_at=at.isoformat())
        self._append_history({"event": "activate", "version": version, "previous": previous,
                              "at": at.isoformat(), "actor": actor, "reason": reason})
        return previous

    def rollback_target(self, exclude: set[str] | None = None) -> str | None:
        """Most recently active version (other than the current one) that is
        still intact and was not itself rolled back."""
        current = self.active_version()
        skip = set(exclude or set()) | {current}
        seen: set[str] = set()
        for record in reversed(self.history()):
            for version in (record.get("previous"), record.get("version")):
                if not version or version in skip or version in seen:
                    continue
                seen.add(version)
                try:
                    meta = self.load_meta(version)
                    if meta.get("status") in TERMINAL_BAD:
                        continue
                    self.verify(version)
                    return version
                except StoreError:
                    continue
        return None

    def rollback(self, *, at: datetime, actor: str, reason: str, to: str | None = None) -> tuple[str | None, str]:
        """Switch back to ``to`` (or the previous good version). The version
        being left is marked rolled_back. Returns (from_version, to_version)."""
        current = self.active_version()
        if to is not None:
            meta = self.load_meta(to)  # raises if unknown
            if to == current:
                raise StoreError(f"{to} is already active")
            if meta.get("status") in {"rejected", "candidate"}:
                raise StoreError(f"{to} has status '{meta.get('status')}' and cannot be activated")
            self.verify(to)
            target = to
        else:
            target = self.rollback_target()
            if target is None:
                raise StoreError("no previous healthy version is available to roll back to")
        atomic_write_json(self.active_file, {"version": target, "previous": current,
                                             "activated_at": at.isoformat(), "actor": actor})
        if current and self.exists(current):
            self.set_status(current, "rolled_back", rolled_back_at=at.isoformat(), rollback_reason=reason)
        self.set_status(target, "active", activated_at=at.isoformat())
        self._append_history({"event": "rollback", "version": target, "previous": current,
                              "at": at.isoformat(), "actor": actor, "reason": reason})
        return current, target

    # ------------------------------------------------------------------ retention
    def prune(self, keep: int, protect: set[str]) -> list[str]:
        """Delete old versions beyond ``keep``, never touching protected ones
        (active, pending, rollback targets)."""
        versions = self.list_versions()
        recent_activated = [r.get("version") for r in self.history()][-3:]
        protected = set(protect) | {self.active_version()} | set(recent_activated)
        removable = [v for v in versions[:-keep] if v not in protected] if len(versions) > keep else []
        for version in removable:
            shutil.rmtree(self.path(version), ignore_errors=True)
        return removable
