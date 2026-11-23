"""Versioning, rollback and the quality gate."""

import json

import pytest
from conftest import FAQ_CSV

from kb_pipeline.config import QualityConfig
from kb_pipeline.documents import KBEntry
from kb_pipeline.errors import StoreError
from kb_pipeline.quality import EvalResult, QualityGate
from kb_pipeline.security import InjectionDetector, PIIMasker


def _activate_next(project, name, text):
    project.write(name, text)
    report = project.pipeline.run_update()
    assert report.status == "activated", report.to_dict()
    project.clock.advance(minutes=6)
    project.pipeline.tick()
    return report.candidate_version


# ----------------------------------------------------------------- versions

def test_each_update_creates_an_immutable_checksummed_version(live_project):
    v2 = _activate_next(live_project, "app.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n")
    store = live_project.pipeline.store
    assert store.list_versions() == ["v000001", "v000002"] and v2 == "v000002"
    meta1, meta2 = store.load_meta("v000001"), store.load_meta("v000002")
    assert meta1["status"] == "retired" and meta2["status"] == "active"
    assert meta2["parent"] == "v000001" and meta2["entry_count"] == 13
    assert set(meta2["checksums"]) == {"entries.jsonl", "source_entries.jsonl", "index.json", "manifest.json"}
    store.verify("v000001")
    assert len(store.load_entries("v000001")) == 12  # old version untouched


def test_tampered_version_is_detected(live_project):
    path = live_project.pipeline.store.path("v000001") / "entries.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("30 days", "300 days"), encoding="utf-8")
    with pytest.raises(StoreError, match="checksum mismatch"):
        live_project.pipeline.store.verify("v000001")


def test_manual_rollback_to_previous_version(live_project):
    _activate_next(live_project, "app.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n")
    result = live_project.pipeline.rollback(actor="admin")
    assert result["from"] == "v000002" and result["to"] == "v000001"
    assert live_project.active() == "v000001"
    assert live_project.pipeline.store.load_meta("v000002")["status"] == "rolled_back"
    history = [h["event"] for h in live_project.pipeline.store.history()]
    assert history == ["activate", "activate", "rollback"]


def test_rollback_to_specific_version(live_project):
    _activate_next(live_project, "a.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n")
    _activate_next(live_project, "b.csv", "question,answer\nWhere are you located?,\"We are fully online.\"\n")
    live_project.pipeline.rollback(actor="admin", to="v000001")
    assert live_project.active() == "v000001"


def test_rollback_without_previous_version_fails(live_project):
    with pytest.raises(StoreError, match="no previous"):
        live_project.pipeline.rollback(actor="admin")


def test_rejected_version_cannot_be_rolled_back_to(live_project):
    live_project.write("faq.csv", FAQ_CSV.replace("Yes, we offer a full refund within 30 days of purchase.",
                                                  "Refunds are not possible."))
    report = live_project.pipeline.run_update()
    assert report.status == "rejected"
    with pytest.raises(StoreError, match="cannot be activated"):
        live_project.pipeline.rollback(actor="admin", to=report.candidate_version)


def test_old_versions_are_pruned_but_active_and_rollback_target_kept(live_project):
    live_project.cfg.versions.keep = 2
    for i in range(4):
        _activate_next(live_project, f"x{i}.csv", f"question,answer\nIs batch {i} open?,\"Yes, batch {i} is open.\"\n")
    versions = live_project.pipeline.store.list_versions()
    assert live_project.active() in versions
    assert live_project.pipeline.store.rollback_target() in versions
    assert len(versions) <= 3


# ----------------------------------------------------------------- quality gate

def test_update_with_wrong_answers_is_rejected(live_project):
    bad = FAQ_CSV.replace("Yes, we offer a full refund within 30 days of purchase.", "Sorry, we never give refunds.") \
                 .replace("The course covers Python, SQL, Excel and Power BI.", "The course covers Java only.")
    live_project.write("faq.csv", bad)
    report = live_project.pipeline.run_update()
    assert report.status == "rejected"
    assert any("accuracy" in r for r in report.gate["reasons"])
    assert live_project.active() == "v000001"
    assert live_project.pipeline.store.load_meta(report.candidate_version)["status"] == "rejected"


def test_update_removing_most_content_is_rejected(live_project):
    live_project.write("faq.csv", "question,answer\nIs there a free trial?,\"Many introductory videos are free.\"\n")
    report = live_project.pipeline.run_update()
    assert report.status == "rejected"
    assert any("entries" in r or "accuracy" in r for r in report.gate["reasons"])


def test_update_that_answers_out_of_scope_questions_is_rejected(live_project):
    live_project.write("trivia.csv", "question,answer\nWhat is the capital of France?,\"Paris.\"\n")
    report = live_project.pipeline.run_update()
    assert report.status == "rejected"
    failed = {i["id"] for i in report.gate["candidate"]["failed_items"]}
    assert "oos-capital" in failed


def test_gate_rejects_accuracy_or_grounding_regression():
    gate = QualityGate(QualityConfig(min_accuracy=0.5, min_grounding=0.5, max_regression=0.0, min_entries=1),
                       PIIMasker(), InjectionDetector())
    entries = [KBEntry("e1", "Q?", "A.", "s", 1)]
    base = EvalResult(accuracy=0.9, grounding=1.0, total=10, answered=9, mean_confidence=0.7, max_latency_ms=1)
    worse_acc = EvalResult(accuracy=0.8, grounding=1.0, total=10, answered=9, mean_confidence=0.7, max_latency_ms=1)
    worse_grounding = EvalResult(accuracy=0.9, grounding=0.9, total=10, answered=9, mean_confidence=0.7, max_latency_ms=1)
    same = EvalResult(accuracy=0.9, grounding=1.0, total=10, answered=9, mean_confidence=0.7, max_latency_ms=1)
    assert not gate.check(worse_acc, base, entries).approved
    assert "grounding would drop" in gate.check(worse_grounding, base, entries).reasons[0]
    assert gate.check(same, base, entries).approved
    leaky = [KBEntry("e2", "Contact?", "Mail ravi.kumar@gmail.com", "s", 1)]
    assert any("sensitive" in r for r in gate.check(same, base, leaky).reasons)


def test_generated_answers_must_be_grounded(project):
    def hallucinating_generator(*, system, question, contexts):
        return "Absolutely, our platinum membership includes free flights and a luxury cruise."

    project.pipeline.generator = hallucinating_generator
    report = project.pipeline.run_update()
    # ungrounded generations are replaced by the stored answer, so grounding stays intact
    assert report.status == "activated"
    engine = project.pipeline.load_engine(project.active())
    answer = engine.answer("Do you offer a refund?")
    assert answer.fallback and "30 days" in answer.text


def test_resubmitting_rolled_back_content_is_rejected(live_project):
    live_project.write("app.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n")
    live_project.faults.set("health_check", 1)
    report = live_project.pipeline.run_update()
    assert report.status == "rolled_back"
    live_project.clock.advance(days=1)
    again = live_project.pipeline.run_update()
    assert again.status == "rejected"
    assert any("rolled back" in r for r in again.gate["reasons"])


def test_update_report_is_logged(live_project):
    lines = (live_project.data / "logs" / "updates.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["status"] == "activated"
