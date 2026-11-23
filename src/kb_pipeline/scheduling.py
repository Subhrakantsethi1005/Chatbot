"""Daily update schedule, retry back-off and the maintenance window."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from .config import ALL_DAYS
from .errors import ConfigError


def _at(day: date, at: time, like: datetime) -> datetime:
    return datetime.combine(day, at, tzinfo=like.tzinfo)


class MaintenanceWindow:
    """A daily window such as 01:00-05:00. Windows may cross midnight
    (22:00-02:00); ``days`` names the weekday on which the window *opens*."""

    def __init__(self, start: time, end: time, days: frozenset[int] = ALL_DAYS) -> None:
        if start == end:
            raise ConfigError("maintenance window start and end must differ")
        self.start, self.end, self.days = start, end, frozenset(days)

    @property
    def crosses_midnight(self) -> bool:
        return self.start > self.end

    def bounds(self, now: datetime) -> tuple[datetime, datetime] | None:
        """(opened_at, closes_at) of the window containing ``now``, else None."""
        t = now.time().replace(tzinfo=None)
        today = now.date()
        if not self.crosses_midnight:
            if self.start <= t < self.end and today.weekday() in self.days:
                return _at(today, self.start, now), _at(today, self.end, now)
            return None
        if t >= self.start and today.weekday() in self.days:
            return _at(today, self.start, now), _at(today + timedelta(days=1), self.end, now)
        yesterday = today - timedelta(days=1)
        if t < self.end and yesterday.weekday() in self.days:
            return _at(yesterday, self.start, now), _at(today, self.end, now)
        return None

    def contains(self, now: datetime) -> bool:
        return self.bounds(now) is not None

    def next_open(self, now: datetime) -> datetime:
        """``now`` if the window is open, otherwise the next opening time."""
        if self.contains(now):
            return now
        for offset in range(0, 9):
            day = now.date() + timedelta(days=offset)
            candidate = _at(day, self.start, now)
            if candidate > now and day.weekday() in self.days:
                return candidate
        raise ConfigError("maintenance window never opens (check maintenance_window.days)")

    def describe(self) -> str:
        return f"{self.start:%H:%M}-{self.end:%H:%M}"


class UpdateSchedule:
    """Run an update every day (or on selected weekdays) at a fixed time."""

    def __init__(self, at: time, days: frozenset[int] = ALL_DAYS) -> None:
        self.at, self.days = at, frozenset(days)

    def latest_occurrence(self, now: datetime) -> datetime | None:
        for offset in range(0, 8):
            day = now.date() - timedelta(days=offset)
            occurrence = _at(day, self.at, now)
            if occurrence <= now and day.weekday() in self.days:
                return occurrence
        return None

    def next_occurrence(self, now: datetime) -> datetime:
        for offset in range(0, 9):
            day = now.date() + timedelta(days=offset)
            occurrence = _at(day, self.at, now)
            if occurrence > now and day.weekday() in self.days:
                return occurrence
        raise ConfigError("update schedule never fires (check schedule.days)")

    def is_due(self, now: datetime, last_handled: datetime | None) -> bool:
        occurrence = self.latest_occurrence(now)
        if occurrence is None:
            return False
        return last_handled is None or occurrence > last_handled


class RetryPolicy:
    """Back-off after consecutive failures: by default the 1st failure is
    retried after 15 minutes, the 2nd after 30, the 3rd after 60; a 4th
    failure gives up until the next scheduled run."""

    def __init__(self, delays_minutes: tuple[float, ...] = (15, 30, 60)) -> None:
        self.delays = tuple(delays_minutes)

    @property
    def max_retries(self) -> int:
        return len(self.delays)

    def next_retry_at(self, failures: int, failed_at: datetime) -> datetime | None:
        if failures < 1 or failures > len(self.delays):
            return None
        return failed_at + timedelta(minutes=self.delays[failures - 1])
