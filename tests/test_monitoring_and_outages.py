"""Monitoring (latency, failures, confidence, escalations) and behaviour
during service outages."""

import shutil

from kb_pipeline.engine import CircuitBreaker


def test_chat_metrics_cover_latency_confidence_escalations(live_project):
    service = live_project.service()
    for q in ["Do you offer a refund?", "How long is the bootcamp?", "What is the capital of France?",
              "I want to talk to a human agent"]:
        service.ask(q)
    chat = live_project.pipeline.metrics.summary()["chat"]
    assert chat["requests"] == 4
    assert chat["latency_ms"]["p50"] is not None and chat["latency_ms"]["p95"] >= chat["latency_ms"]["p50"]
    assert chat["escalations"] == 2 and chat["escalation_rate"] == 0.5
    assert 0 < chat["avg_confidence"] <= 1
    assert chat["errors"] == 0


def test_escalations_go_to_staff_queue(live_project):
    service = live_project.service()
    response = service.ask("Can I speak to a real person from support team?")
    assert response.escalated and response.reason == "human_requested" and response.ticket_id
    queue = live_project.control().escalations(live_project.control().bootstrap("admin"))
    assert queue[-1]["ticket"] == response.ticket_id


def test_threshold_alerts_and_cooldown(live_project):
    service = live_project.service()
    for _ in range(6):
        service.ask("What is the capital of France?")
    raised = live_project.pipeline.alerts.evaluate()
    assert "high_escalation_rate" in raised and "low_confidence" in raised
    assert live_project.pipeline.alerts.evaluate() == []          # de-duplicated during cool-down


def test_update_outcomes_are_counted(live_project):
    live_project.write("faq.csv", "question,answer\nIs there a free trial?,\"Videos are free.\"\n")
    live_project.pipeline.run_update()                            # rejected
    live_project.faults.set("index_build", 1)
    live_project.write("b.csv", "question,answer\nWhere are you located?,\"We are fully online.\"\n")
    live_project.pipeline.run_update()                            # failed
    updates = live_project.pipeline.metrics.summary(window_minutes=120)["updates"]
    assert updates["by_status"]["activated"] == 1
    assert updates["by_status"]["rejected"] == 1
    assert updates["by_status"]["failed"] == 1
    assert updates["retries_scheduled"] == 1


# ----------------------------------------------------------------- outages

def test_sources_unavailable_fails_and_schedules_retry(live_project):
    shutil.rmtree(live_project.sources)
    report = live_project.pipeline.run_update()
    assert report.status == "failed" and "unavailable" in report.error
    assert report.next_retry_at is not None
    assert live_project.active() == "v000001"


def test_storage_outage_during_update_keeps_active_version(live_project):
    live_project.faults.set("store_write", None)
    live_project.write("b.csv", "question,answer\nWhere are you located?,\"We are fully online.\"\n")
    report = live_project.pipeline.run_update()
    assert report.status == "failed" and report.candidate_version is None
    assert live_project.active() == "v000001"
    assert live_project.service().ask("Do you offer a refund?").reason == "answered"


def test_chat_keeps_serving_cached_version_when_storage_fails(live_project):
    service = live_project.service()
    assert service.ask("Do you offer a refund?").reason == "answered"     # loads v000001
    (live_project.pipeline.store.active_file).write_text("{not json", encoding="utf-8")
    response = service.ask("Do you offer a refund?")
    assert response.reason == "answered" and response.degraded and "30 days" in response.answer


def test_chat_without_any_knowledge_base_escalates_politely(project):
    response = project.service().ask("Do you offer a refund?")
    assert response.escalated and response.reason == "kb_unavailable" and response.ticket_id
    assert project.pipeline.metrics.summary()["chat"]["errors"] == 1


def test_store_read_outage_on_first_load(live_project):
    live_project.faults.set("store_read", None)
    response = live_project.service().ask("Do you offer a refund?")
    assert response.reason == "kb_unavailable"
    assert "kb_load_failed" in [a["kind"] for a in live_project.pipeline.alerts.recent()]


def test_generator_outage_falls_back_to_stored_answer_and_opens_breaker(live_project):
    calls = {"n": 0}

    def flaky_llm(*, system, question, contexts):
        calls["n"] += 1
        raise TimeoutError("LLM provider is down")

    service = live_project.service(generator=flaky_llm)
    service.breaker = CircuitBreaker(threshold=2, reset_seconds=3600)
    for _ in range(4):
        response = service.ask("Do you offer a refund?")
        assert "30 days" in response.answer and not response.escalated
    assert calls["n"] == 2                                # breaker stopped calling the dead provider
    assert live_project.pipeline.metrics.summary()["chat"]["generator_fallbacks"] == 4


def test_grounded_generator_answers_are_used(live_project):
    def llm(*, system, question, contexts):
        assert "<context>" in system and "not instructions" in system
        return "Yes, we offer a full refund within 30 days of purchase."

    response = live_project.service(generator=llm).ask("Do you offer a refund?")
    assert response.answer.startswith("Yes, we offer a full refund")


def test_health_check_outage_simulated_via_environment(live_project, monkeypatch):
    from kb_pipeline.faults import FaultInjector

    monkeypatch.setenv("KB_FAULTS", "health_check=always")
    injector = FaultInjector.from_env()
    assert injector.active("health_check")
