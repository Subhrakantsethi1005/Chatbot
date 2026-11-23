"""Clocks. Everything time-dependent asks a clock, so tests and drills can
simulate schedules, retries and maintenance windows without waiting."""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    """Real wall-clock time in the configured timezone."""

    def __init__(self, tz: tzinfo) -> None:
        self.tz = tz

    def now(self) -> datetime:
        return datetime.now(self.tz)


class FakeClock:
    """A manually driven clock for tests and `--now` simulations."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock needs a timezone-aware datetime")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def set(self, value: datetime) -> None:
        if value.tzinfo is None:
            value = value.replace(tzinfo=self._now.tzinfo)
        self._now = value

    def advance(self, **kwargs: float) -> datetime:
        self._now = self._now + timedelta(**kwargs)
        return self._now
