"""Incremental processing: only new or modified documents are processed."""

import json

from kb_pipeline.config import IngestionConfig
from kb_pipeline.documents import load_document, parse_text


def test_first_update_builds_and_activates_inside_window(project):
    report = project.pipeline.run_update()
    assert report.status == "activated"
    assert report.candidate_version == "v000001"
    assert report.processed == ["faq.csv"]
    assert report.entry_count == 12
    assert project.active() == "v000001"
    assert report.gate["candidate"]["accuracy"] == 1.0


def test_unchanged_sources_create_no_new_version(live_project):
    report = live_project.pipeline.run_update()
    assert report.status == "no_changes"
    assert report.processed == []
    assert report.unchanged == ["faq.csv"]
    assert live_project.pipeline.store.list_versions() == ["v000001"]


def test_only_new_document_is_processed(live_project):
    live_project.write("extra.csv", "question,answer\nDo you have a mobile app?,\"Yes, our app is on Android and iOS.\"\n")
    report = live_project.pipeline.run_update()
    assert report.processed == ["extra.csv"]
    assert report.unchanged == ["faq.csv"]
    assert report.status == "activated"
    assert "Do you have a mobile app?" in live_project.active_questions()
    assert len(live_project.active_questions()) == 13


def test_only_modified_document_is_processed(live_project):
    live_project.write("extra.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n")
    live_project.pipeline.run_update()
    live_project.clock.advance(minutes=6)
    live_project.pipeline.tick()
    path = live_project.write("extra.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android and iOS.\"\n")
    report = live_project.pipeline.run_update()
    assert report.processed == ["extra.csv"]
    assert sorted(report.unchanged) == ["faq.csv"]
    answers = {e.answer for e in live_project.pipeline.store.load_entries(live_project.active())}
    assert "Yes, on Android and iOS." in answers and "Yes, on Android." not in answers
    assert path.exists()


def test_deleted_document_entries_are_removed(live_project):
    live_project.write("extra.csv", "question,answer\nDo you have a mobile app?,\"Yes, on Android.\"\n")
    live_project.pipeline.run_update()
    live_project.clock.advance(minutes=6)
    live_project.pipeline.tick()
    (live_project.sources / "extra.csv").unlink()
    report = live_project.pipeline.run_update()
    assert report.deleted == ["extra.csv"]
    assert "Do you have a mobile app?" not in live_project.active_questions()


def test_windows_1252_file_is_accepted(live_project):
    text = "question,answer\nWhat’s the batch size?,\"Each batch has 40 learners – small and focused.\"\n"
    live_project.write("cp1252.csv", text, encoding="cp1252")
    report = live_project.pipeline.run_update()
    assert report.status == "activated"
    assert any("cp1252" in w for w in report.warnings)
    assert "What’s the batch size?" in live_project.active_questions()


def test_json_markdown_and_text_formats(live_project):
    live_project.write("faq.json", json.dumps([{"question": "Do you run weekend batches?",
                                                 "answer": "Yes, weekend batches start every month."}]))
    live_project.write("policies.md", "# Can I switch batches?\nYes, you can switch batch once for free.\n")
    live_project.write("notes.txt", "Q: Is there a student discount?\nA: Students get 10% off with a valid ID.\n")
    report = live_project.pipeline.run_update()
    assert sorted(report.processed) == ["faq.json", "notes.txt", "policies.md"]
    questions = live_project.active_questions()
    assert {"Do you run weekend batches?", "Can I switch batches?", "Is there a student discount?"} <= questions


def test_parse_text_paragraphs_become_entries():
    rows = parse_text("Our office is in Bangalore.\nIt opens at 9am.\n\nWe close on Sundays.")
    assert [r.answer for r in rows] == ["Our office is in Bangalore. It opens at 9am.", "We close on Sundays."]


def test_hidden_and_temp_files_are_ignored(live_project):
    live_project.write(".hidden.csv", "question,answer\nx?,y\n")
    live_project.write("~$lock.csv", "question,answer\nx?,y\n")
    report = live_project.pipeline.run_update()
    assert report.status == "no_changes"


def test_load_document_reports_encoding():
    doc = load_document("a.csv", "question,answer\nq?,a\n".encode("utf-8"), IngestionConfig())
    assert doc.encoding == "utf-8-sig" and doc.warnings == []
