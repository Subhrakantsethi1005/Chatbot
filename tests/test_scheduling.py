"""Configurable schedule, retries after 15/30/60 minutes and the
maintenance window - all simulated with a FakeClock."""

from datetime import datetime, time, timedelta, timezone

import pytest
from conftest import OUTSIDE_WINDOW, make_project

from kb_pipeline.config import ConfigError
from kb_pipeline.scheduling import MaintenanceWindow, RetryPolicy, UpdateSchedule

UTC = timezone.utc


def at(hh, mm=0, day=24):
    return datetime(2026, 11, day, hh, mm, tzinfo=UTC)


def update_events(project):
    return [e for e in project.pipeline.events.read() if e["kind"] == "update"]


# ----------------------------------------------------------------- units

def test_maintenance_window_same_day():
    w = MaintenanceWindow(time(1, 0), time(5, 0))
    assert w.contains(at(1, 0)) and w.contains(at(4, 59))
    assert not w.contains(at(0, 59)) and not w.contains(at(5, 0)) and not w.contains(at(14))
    assert w.next_open(at(14)) == at(1, 0, day=25)


def test_maintenance_window_crossing_midnight_and_weekdays():
    w = MaintenanceWindow(time(22, 0), time(2, 0), days=frozenset({5}))  # opens Saturdays only
    saturday, sunday = datetime(2026, 11, 28, 23, 0, tzinfo=UTC), datetime(2026, 11, 29, 1, 0, tzinfo=UTC)
    assert w.contains(saturday) and w.contains(sunday)
    assert not w.contains(datetime(2026, 11, 29, 23, 0, tzinfo=UTC))   # Sunday night: closed
    assert w.next_open(datetime(2026, 11, 24, 12, 0, tzinfo=UTC)) == datetime(2026, 11, 28, 22, 0, tzinfo=UTC)


def test_window_start_equal_end_is_rejected():
    with pytest.raises(ConfigError):
        MaintenanceWindow(time(1, 0), time(1, 0))


def test_retry_policy_15_30_60():
    policy = RetryPolicy((15, 30, 60))
    failed = at(2)
    assert policy.next_retry_at(1, failed) == at(2, 15)
    assert policy.next_retry_at(2, failed) == at(2, 30)
    assert policy.next_retry_at(3, failed) == at(3)
    assert policy.next_retry_at(4, failed) is None


def test_update_schedule_due():
    schedule = UpdateSchedule(time(2, 30))
    assert schedule.latest_occurrence(at(2, 29)) == at(2, 30, day=23)
    assert schedule.is_due(at(2, 30), last_handled=at(2, 30, day=23))
    assert not schedule.is_due(at(3), last_handled=at(2, 30))
    assert schedule.next_occurrence(at(2, 31)) == at(2, 30, day=25)


# ----------------------------------------------------------------- schedule

def test_update_runs_at_configured_time_once_per_day(tmp_path):
    project = make_project(tmp_path, now=at(2, 0) - timedelta(minutes=1), update_time="02:00")
    assert project.pipeline.tick()["actions"] == []
    project.clock.set(at(2, 0))
    result = project.pipeline.tick()
    assert result["actions"][0]["event"] == "update" and result["actions"][0]["status"] == "activated"
    for _ in range(10):
        project.clock.advance(seconds=30)
        project.pipeline.tick()
    assert len(update_events(project)) == 1
    project.clock.set(at(2, 0, day=25))
    project.pipeline.tick()
    assert len(update_events(project)) == 2


def test_custom_update_time_is_respected(tmp_path):
    project = make_project(tmp_path, now=at(3, 0), update_time="03:45")
    project.pipeline.tick()
    assert update_events(project) == []
    project.clock.set(at(3, 45))
    project.pipeline.tick()
    assert len(update_events(project)) == 1


def test_missed_schedule_is_caught_up_once(tmp_path):
    project = make_project(tmp_path, now=at(1, 0), update_time="02:00")
    project.pipeline.tick()
    project.clock.set(at(4, 0))          # daemon was down at 02:00
    project.pipeline.tick()
    project.pipeline.tick()
    assert len(update_events(project)) == 1


# ----------------------------------------------------------------- retries

def test_failed_update_is_retried_after_15_30_and_60_minutes(tmp_path):
    project = make_project(tmp_path, now=at(1, 59), update_time="02:00")
    project.pipeline.tick()
    project.faults.set("index_build", None)  # outage that never recovers
    project.clock.set(at(2, 0))
    project.pipeline.tick()
    retry = project.pipeline.status()["retry"]
    assert retry["failures"] == 1 and retry["next_at"] == at(2, 15).isoformat()

    expected = [(at(2, 15), at(2, 45)), (at(2, 45), at(3, 45))]
    for run_at, next_at in expected:
        project.clock.set(run_at - timedelta(minutes=1))
        project.pipeline.tick()                       # not yet due
        project.clock.set(run_at)
        project.pipeline.tick()
        assert project.pipeline.status()["retry"]["next_at"] == next_at.isoformat()

    project.clock.set(at(3, 45))
    project.pipeline.tick()                           # third retry fails -> give up
    assert project.pipeline.status()["retry"] is None
    statuses = [e["status"] for e in update_events(project)]
    assert statuses == ["failed"] * 4
    kinds = [a["kind"] for a in project.pipeline.alerts.recent(20)]
    assert kinds.count("update_failed") == 3 and "update_failed_permanently" in kinds
    assert project.active() is None


def test_retry_succeeds_after_outage_recovers(tmp_path):
    project = make_project(tmp_path, now=at(1, 59), update_time="02:00")
    project.pipeline.tick()
    project.faults.set("store_write", 2)              # fails twice, then recovers
    project.clock.set(at(2, 0))
    project.pipeline.tick()
    project.clock.set(at(2, 15))
    project.pipeline.tick()
    assert project.pipeline.status()["retry"]["next_at"] == at(2, 45).isoformat()
    project.clock.set(at(2, 45))
    result = project.pipeline.tick()
    assert result["actions"][0]["status"] == "activated"
    assert project.pipeline.status()["retry"] is None
    assert project.active() == "v000001"


def test_rejected_update_is_not_retried(live_project):
    live_project.write("faq.csv", "question,answer\nIs there a free trial?,\"Videos are free.\"\n")
    report = live_project.pipeline.run_update()
    assert report.status == "rejected"
    assert live_project.pipeline.status()["retry"] is None


def test_concurrent_run_is_reported_busy(project):
    project.pipeline.lock.acquire()                   # simulate another process holding the lock
    try:
        assert project.pipeline.tick()["actions"][0]["event"] == "busy"
        report = project.pipeline.run_update()
        assert report.status == "failed" and "in progress" in report.error
        assert project.pipeline.status()["retry"]["failures"] == 1
    finally:
        project.pipeline.lock.release()


# ----------------------------------------------------------------- maintenance window

def test_approved_update_waits_for_the_maintenance_window(tmp_path):
    project = make_project(tmp_path, now=OUTSIDE_WINDOW)
    report = project.pipeline.run_update()
    assert report.status == "pending_activation"
    assert "outside the maintenance window" in report.activation["reason"]
    assert project.active() is None

    for minutes in (60, 600):                         # still outside (15:00, 00:00)
        project.clock.set(OUTSIDE_WINDOW + timedelta(minutes=minutes))
        project.pipeline.tick()
        assert project.active() is None
    project.clock.set(at(1, 0, day=25))               # window opens
    result = project.pipeline.tick()
    assert any(a.get("event") == "activation" and a.get("activated") for a in result["actions"])
    assert project.active() == report.candidate_version


def test_manual_activation_is_refused_outside_window(tmp_path):
    project = make_project(tmp_path, now=OUTSIDE_WINDOW)
    project.pipeline.run_update()
    outcome = project.pipeline.activate_pending(actor="operator")
    assert outcome["activated"] is False and "next opens" in outcome["reason"]
    project.clock.set(at(2, 0, day=25))
    assert project.pipeline.activate_pending(actor="operator")["activated"] is True


def test_newer_approved_update_supersedes_pending_one(tmp_path):
    project = make_project(tmp_path, now=OUTSIDE_WINDOW)
    first = project.pipeline.run_update()
    project.write("app.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n")
    project.clock.advance(minutes=5)
    second = project.pipeline.run_update()
    assert second.status == "pending_activation" and second.base_version == first.candidate_version
    assert project.pipeline.store.load_meta(first.candidate_version)["status"] == "superseded"
    project.clock.set(at(1, 5, day=25))
    project.pipeline.tick()
    assert project.active() == second.candidate_version


def test_simulated_night_with_daemon(tmp_path):
    """23:50 -> 02:20 in 30-second ticks: the 02:00 update runs once, inside
    the 01:00-05:00 window, and passes its 5-minute health watch."""
    project = make_project(tmp_path, now=datetime(2026, 11, 23, 23, 50, tzinfo=UTC))
    plane = project.control()
    token = plane.bootstrap("ops")
    ticks = plane.daemon(token, iterations=int(150 * 60 / 30))
    assert ticks == 300
    assert [e["status"] for e in update_events(project)] == ["activated"]
    assert project.pipeline.store.load_meta(project.active())["health"] == "stable"
