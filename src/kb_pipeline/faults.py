"""Fault injection for outage drills.

Production code calls ``faults.check("<point>")`` at the places where a real
outage would surface. Nothing happens unless a fault is configured, either
in code (tests) or with the ``KB_FAULTS`` environment variable, e.g.::

    KB_FAULTS="index_build=2,health_check=always"

Points: ``source_read``, ``index_build``, ``store_write``, ``store_read``,
``health_check``, ``generator``.
"""

from __future__ import annotations

import os

from .errors import TransientError

KNOWN_POINTS = {"source_read", "index_build", "store_write", "store_read", "health_check", "generator"}


class FaultInjector:
    def __init__(self, spec: dict[str, int | None] | None = None) -> None:
        # value None == fail forever, int == fail that many more times
        self._remaining: dict[str, int | None] = dict(spec or {})
        unknown = set(self._remaining) - KNOWN_POINTS
        if unknown:
            raise ValueError(f"unknown fault point(s): {', '.join(sorted(unknown))}")

    @classmethod
    def from_env(cls, variable: str = "KB_FAULTS") -> "FaultInjector":
        return cls.parse(os.environ.get(variable, ""))

    @classmethod
    def parse(cls, text: str) -> "FaultInjector":
        spec: dict[str, int | None] = {}
        for part in filter(None, (p.strip() for p in (text or "").split(","))):
            name, _, count = part.partition("=")
            count = count.strip().lower()
            spec[name.strip()] = None if count in ("", "always", "forever") else int(count)
        return cls(spec)

    def set(self, point: str, times: int | None = None) -> None:
        if point not in KNOWN_POINTS:
            raise ValueError(f"unknown fault point: {point}")
        self._remaining[point] = times

    def clear(self, point: str | None = None) -> None:
        if point is None:
            self._remaining.clear()
        else:
            self._remaining.pop(point, None)

    def active(self, point: str) -> bool:
        return point in self._remaining

    def check(self, point: str) -> None:
        if point not in self._remaining:
            return
        remaining = self._remaining[point]
        if remaining is not None:
            if remaining <= 1:
                del self._remaining[point]
            else:
                self._remaining[point] = remaining - 1
        raise TransientError(f"simulated outage at '{point}'")
