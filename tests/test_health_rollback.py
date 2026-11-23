"""Automatic rollback when health checks fail within five minutes."""

from datetime import timedelta


def _stage_v2(project):
    project.write("app.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n")
    report = project.pipeline.run_update()
    assert report.status in ("activated", "rolled_back"), report.to_dict()
    return report


def test_failed_first_health_check_rolls_back_immediately(live_project):
    live_project.faults.set("health_check", 1)
    report = _stage_v2(live_project)
    assert report.status == "rolled_back"
    assert live_project.active() == "v000001"
    assert live_project.pipeline.store.load_meta("v000002")["status"] == "rolled_back"
    kinds = [a["kind"] for a in live_project.pipeline.alerts.recent()]
    assert "automatic_rollback" in kinds


def test_health_failure_at_three_minutes_rolls_back(live_project):
    _stage_v2(live_project)
    assert live_project.active() == "v000002"
    for _ in range(5):                                       # 2.5 healthy minutes
        live_project.clock.advance(seconds=30)
        live_project.pipeline.tick()
    assert live_project.active() == "v000002"
    live_project.faults.set("health_check", None)            # outage starts
    live_project.clock.advance(seconds=30)
    result = live_project.pipeline.tick()
    assert result["actions"][0]["event"] == "rollback"
    assert live_project.active() == "v000001"
    events = [e for e in live_project.pipeline.events.read() if e["kind"] == "rollback"]
    assert events[-1]["automatic"] is True and events[-1]["from_version"] == "v000002"


def test_corrupted_files_after_activation_trigger_rollback(live_project):
    _stage_v2(live_project)
    index = live_project.pipeline.store.path("v000002") / "index.json"
    index.write_text("{}", encoding="utf-8")
    live_project.clock.advance(seconds=30)
    live_project.pipeline.tick()
    assert live_project.active() == "v000001"


def test_high_live_error_rate_triggers_rollback(live_project):
    _stage_v2(live_project)
    for _ in range(6):
        live_project.pipeline.metrics.record_chat(latency_ms=5, confidence=0, escalated=True, blocked=False,
                                                  error=True, version="v000002", reason="engine_error")
    live_project.clock.advance(seconds=30)
    live_project.pipeline.tick()
    assert live_project.active() == "v000001"


def test_no_automatic_rollback_after_watch_period(live_project):
    _stage_v2(live_project)
    for _ in range(10):
        live_project.clock.advance(seconds=30)
        live_project.pipeline.tick()
    live_project.clock.advance(seconds=31)                   # 5 min 31 s after activation
    live_project.pipeline.tick()
    assert live_project.pipeline.status()["health_watch"] is None
    assert live_project.pipeline.store.load_meta("v000002")["health"] == "stable"
    live_project.faults.set("health_check", None)
    live_project.clock.advance(minutes=1)
    live_project.pipeline.tick()
    assert live_project.active() == "v000002"                # watch is over: alert only, no rollback


def test_health_check_reports_slow_answers(live_project):
    live_project.cfg.health.max_latency_ms = 0.000001
    result = live_project.pipeline.health.check("v000001")
    assert not result.ok and "took" in result.error


def test_failed_health_without_previous_version_alerts(project):
    project.faults.set("health_check", 1)
    report = project.pipeline.run_update()
    assert report.activation["first_health_check"]["event"] == "rollback_failed"
    assert "rollback_impossible" in [a["kind"] for a in project.pipeline.alerts.recent()]


def test_manual_rollback_cancels_health_watch(live_project):
    _stage_v2(live_project)
    live_project.pipeline.rollback(actor="admin")
    assert live_project.pipeline.status()["health_watch"] is None
    live_project.clock.advance(seconds=30)
    assert live_project.pipeline.tick()["actions"] == []
    assert live_project.active() == "v000001"


def test_stale_watch_does_not_block_updates_run_without_daemon(project):
    """CLI-only operation (no ticks): a finished watch period must not block
    the next activation."""
    assert project.pipeline.run_update().status == "activated"
    project.clock.advance(minutes=30)                       # nobody called tick()
    project.write("app.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n")
    report = project.pipeline.run_update()
    assert report.status == "activated", report.activation
    assert project.pipeline.store.load_meta("v000001")["health"] == "stable"


def test_activation_waits_while_health_watch_runs(live_project):
    _stage_v2(live_project)
    live_project.write("b.csv", "question,answer\nWhere are you located?,\"We are fully online.\"\n")
    report = live_project.pipeline.run_update()
    assert report.status == "pending_activation"
    assert "health watch" in report.activation["reason"]
    live_project.clock.advance(minutes=5, seconds=30)
    live_project.pipeline.tick()          # watch finishes
    live_project.clock.advance(seconds=30)
    live_project.pipeline.tick()          # pending activates
    assert live_project.active() == report.candidate_version
    assert timedelta(minutes=5) == timedelta(minutes=live_project.cfg.health.watch_minutes)
