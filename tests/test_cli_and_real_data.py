"""End-to-end through the CLI, and the shipped dataset + eval set."""

import json
import re
import shutil
from pathlib import Path

from conftest import make_project

from kb_pipeline.cli import main
from kb_pipeline.clock import FakeClock
from kb_pipeline.config import load_config
from kb_pipeline.pipeline import KnowledgeBasePipeline

ROOT = Path(__file__).resolve().parents[1]


def _token(output: str) -> str:
    return re.search(r"(kb_[A-Za-z0-9_-]{20,})", output).group(1)


def test_cli_end_to_end(tmp_path, capsys):
    project = make_project(tmp_path)
    config = str(project.root / "config.toml")
    now = ["--config", config, "--now", "2026-11-24T01:30:00+00:00"]

    assert main([*now, "bootstrap", "--name", "root"]) == 0
    admin = _token(capsys.readouterr().out)
    assert main([*now, "--token", admin, "user-add", "--name", "dash", "--role", "viewer"]) == 0
    viewer = _token(capsys.readouterr().out)

    assert main([*now, "--token", admin, "update"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "activated"

    assert main([*now, "--token", viewer, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["active"]["version"] == "v000001"
    assert main([*now, "--token", viewer, "update"]) == 2
    capsys.readouterr()

    assert main([*now, "ask", "Do you offer a refund?"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert "30 days" in answer["answer"]

    assert main([*now, "--token", viewer, "metrics", "--window", "600"]) == 0
    assert json.loads(capsys.readouterr().out)["chat"]["requests"] >= 1

    (project.sources / "faq.csv").write_text("question,answer\nIs there a free trial?,\"Free videos.\"\n",
                                             encoding="utf-8")
    assert main([*now, "--token", admin, "update"]) == 3        # rejected by the quality gate
    capsys.readouterr()
    assert main([*now, "--faults", "index_build=1", "--token", admin, "update"]) == 4   # simulated outage
    failed = json.loads(capsys.readouterr().out)
    assert failed["next_retry_at"].startswith("2026-11-24T01:45")


def test_cli_daemon_simulates_a_night(tmp_path, capsys):
    project = make_project(tmp_path)
    config = str(project.root / "config.toml")
    assert main(["--config", config, "bootstrap"]) == 0
    admin = _token(capsys.readouterr().out)
    assert main(["--config", config, "--token", admin, "--now", "2026-11-24T01:58:00+00:00",
                 "daemon", "--iterations", "20"]) == 0
    output = capsys.readouterr().out
    assert '"status": "activated"' in output


def test_shipped_dataset_passes_quality_gate(tmp_path):
    """The real FAQ sheet + config/eval_set.jsonl must produce an approved KB."""
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
    cfg = load_config(ROOT / "config" / "pipeline.toml")
    cfg.paths.sources_dir = tmp_path / "dataset"
    cfg.paths.data_dir = tmp_path / "kb_data"
    from datetime import datetime

    clock = FakeClock(datetime(2026, 11, 24, 2, 0, tzinfo=cfg.tz))
    pipeline = KnowledgeBasePipeline(cfg, clock=clock)
    report = pipeline.run_update()
    assert report.status == "activated", report.gate
    assert report.entry_count == 76
    assert report.gate["candidate"]["accuracy"] >= cfg.quality.min_accuracy
    assert report.gate["candidate"]["grounding"] == 1.0
    assert not report.quarantined and not report.pii_masked


def test_shipped_config_loads_and_rejects_typos(tmp_path):
    cfg = load_config(ROOT / "config" / "pipeline.toml")
    assert cfg.schedule.retry_delays_minutes == (15, 30, 60)
    assert cfg.health.watch_minutes == 5
    bad = tmp_path / "bad.toml"
    bad.write_text("[schedule]\nretry_delay = [1]\n", encoding="utf-8")
    import pytest

    from kb_pipeline.errors import ConfigError
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(bad)
