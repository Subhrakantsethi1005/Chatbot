"""Duplicate documents and duplicate / near-duplicate / conflicting entries."""

from conftest import FAQ_CSV

from kb_pipeline.dedupe import dedupe_entries
from kb_pipeline.documents import KBEntry, make_entry_id


def _entry(source, question, answer, row=1):
    return KBEntry(make_entry_id(source, question, answer), question, answer, source, row)


def test_exact_duplicate_document_is_detected_and_skipped(live_project):
    live_project.write("faq_copy.csv", FAQ_CSV)
    report = live_project.pipeline.run_update()
    assert report.duplicate_documents == [{"path": "faq_copy.csv", "duplicate_of": "faq.csv"}]
    assert report.status == "no_changes"
    assert live_project.pipeline.store.list_versions() == ["v000001"]
    # the copy is remembered and not reprocessed on the next run
    again = live_project.pipeline.run_update()
    assert again.processed == []
    assert again.skipped == [{"path": "faq_copy.csv", "status": "duplicate", "duplicate_of": "faq.csv"}]


def test_duplicate_document_with_other_encoding_and_line_endings(live_project):
    live_project.write_bytes("faq_windows.csv", FAQ_CSV.replace("\n", "\r\n").encode("cp1252"))
    report = live_project.pipeline.run_update()
    assert report.duplicate_documents[0]["duplicate_of"] == "faq.csv"


def test_duplicate_of_deleted_original_is_ingested(live_project):
    live_project.write("faq_copy.csv", FAQ_CSV)
    live_project.pipeline.run_update()
    (live_project.sources / "faq.csv").unlink()
    report = live_project.pipeline.run_update()
    assert "faq_copy.csv" in report.processed
    assert report.deleted == ["faq.csv"]
    assert len(live_project.active_questions()) == 12


def test_duplicate_entries_across_files_are_dropped(live_project):
    live_project.write("more.csv", "question,answer\n"
                       "Do you offer a refund?,\"Yes, we offer a full refund within 30 days of purchase.\"\n"
                       "do you offer a REFUND,\"Yes we offer a full refund within 30 days of purchase\"\n"
                       "Do you have a mobile app?,\"Yes, on Android.\"\n")
    report = live_project.pipeline.run_update()
    kinds = sorted(d["kind"] for d in report.duplicate_entries)
    assert kinds == ["exact", "exact"]
    entries = live_project.pipeline.store.load_entries(live_project.active())
    assert sum(e.question.lower().startswith("do you offer a refund") for e in entries) == 1
    assert len(entries) == 13


def test_near_duplicate_entry_is_dropped():
    a = _entry("a.csv", "Do you provide any virtual internship?", "Yes, every learner can join one.")
    b = _entry("b.csv", "Do you provide any virtual internship ?!", "Yes every learner can join one")
    c = _entry("b.csv", "Do you provide any virtual internship program?", "Yes, every learner can join one.")
    result = dedupe_entries([a, b, c], threshold=0.8)
    assert [e.id for e in result.entries] == [a.id]
    assert {d.kind for d in result.duplicates} == {"exact", "near"}


def test_similar_questions_with_different_answers_are_kept():
    a = _entry("a.csv", "Is there any prerequisite for taking this bootcamp?", "You need a laptop.")
    b = _entry("a.csv", "Is there any prerequisite for taking this course?", "No coding background needed.")
    result = dedupe_entries([a, b])
    assert len(result.entries) == 2 and not result.duplicates


def test_conflicting_answer_newer_source_wins():
    old = _entry("old.csv", "Is there an EMI option?", "No.")
    new = _entry("new.csv", "Is there an EMI option?", "Yes, 3 and 6 month EMI plans.")
    result = dedupe_entries([old, new], source_rank={"old.csv": 1, "new.csv": 2})
    assert [e.answer for e in result.entries] == ["Yes, 3 and 6 month EMI plans."]
    assert result.conflicts[0].dropped_source == "old.csv"


def test_dropped_duplicate_returns_when_original_file_is_deleted(live_project):
    live_project.write("a.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n")
    live_project.pipeline.run_update()
    live_project.clock.advance(minutes=6)
    live_project.pipeline.tick()
    live_project.write("b.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n"
                                "Where are you located?,\"We are fully online.\"\n")
    report = live_project.pipeline.run_update()
    assert any(d["dropped_source"] == "b.csv" for d in report.duplicate_entries)
    live_project.clock.advance(minutes=6)
    live_project.pipeline.tick()
    (live_project.sources / "a.csv").unlink()
    live_project.pipeline.run_update()
    assert "Do you have a mobile app?" in live_project.active_questions()
