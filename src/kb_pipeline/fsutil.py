"""Small filesystem helpers: atomic JSON writes and an inter-process lock.

Windows (and OneDrive) can briefly hold a file open, which makes
``os.replace`` fail with PermissionError; writes are retried a few times
before surfacing as a TransientError.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import LockHeldError, TransientError


def atomic_write_text(path: Path, text: str, *, attempts: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            return
        except PermissionError as exc:  # file briefly locked by a reader/sync client
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
        except OSError as exc:
            last_error = exc
            break
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    raise TransientError(f"could not write {path}: {last_error}")


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


class FileLock:
    """Exclusive lock using O_CREAT|O_EXCL. A lock older than ``stale_after``
    seconds (a crashed run) is broken automatically."""

    def __init__(self, path: Path, *, stale_after: float = 3600) -> None:
        self.path = path
        self.stale_after = stale_after
        self._held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.stale_after:
                    try:
                        self.path.unlink()
                    except OSError:
                        pass
                    continue
                raise LockHeldError(f"another pipeline run is in progress (lock {self.path})")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "at": datetime.now(timezone.utc).isoformat()}))
            self._held = True
            return
        raise LockHeldError(f"could not acquire lock {self.path}")

    def release(self) -> None:
        if self._held:
            try:
                self.path.unlink()
            except OSError:
                pass
            self._held = False

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
