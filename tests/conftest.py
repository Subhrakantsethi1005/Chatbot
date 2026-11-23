"""Shared fixtures: an isolated project (sources, config, eval set, data
folder) driven by a FakeClock, so schedules and windows are simulated."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kb_pipeline.clock import FakeClock
from kb_pipeline.config import PipelineConfig, load_config
from kb_pipeline.control import ControlPlane
from kb_pipeline.faults import FaultInjector
from kb_pipeline.pipeline import KnowledgeBasePipeline
from kb_pipeline.service import ChatService

FAQ_CSV = """question,answer
Do you offer a refund?,"Yes, we offer a full refund within 30 days of purchase."
Is there an EMI option?,"No, we do not offer EMI payments at the moment."
Do you provide virtual internships?,"Yes, every learner can join a virtual internship after finishing the course."
How long is the bootcamp?,"The bootcamp takes about 3 months at 2-3 hours per day."
Can I access the course for lifetime?,"Yes, you get lifetime access to all recorded lectures."
Do I need a laptop?,"Yes, you need a laptop with at least 8GB RAM to follow the projects."
Is the course self paced?,"Yes, the course is fully self paced so you can learn alongside a job."
Do you provide job assistance?,"Yes, we help with resume reviews and mock interviews."
Which tools are covered in the course?,"The course covers Python, SQL, Excel and Power BI."
How can I contact support?,"Post your question in our discord server and a mentor will reply within a day."
Can I get a certificate?,"Yes, you receive a certificate after completing the final project."
Is there a free trial?,"Many introductory videos are free on our YouTube channel."
"""

EVAL_ITEMS = [
    {"id": "refund", "question": "Can I get a refund on my purchase?", "expected_keywords": ["refund", "30 days"]},
    {"id": "emi", "question": "Can I pay with an EMI option?", "expected_questions": ["Is there an EMI option?"],
     "expected_keywords": ["no"]},
    {"id": "internship", "question": "Do you provide a virtual internship?", "expected_keywords": ["internship"]},
    {"id": "duration", "question": "How long does the bootcamp take?", "expected_keywords": ["3 months"]},
    {"id": "laptop", "question": "Do I need my own laptop?", "expected_keywords": ["laptop"]},
    {"id": "job", "question": "Do you provide any job assistance?", "expected_keywords": ["resume"]},
    {"id": "tools", "question": "Which tools does the course cover?", "expected_keywords": ["python"]},
    {"id": "support", "question": "How do I contact support?", "expected_keywords": ["discord"]},
    {"id": "certificate", "question": "Will I get a certificate?", "expected_keywords": ["certificate"]},
    {"id": "oos-capital", "question": "What is the capital of France?", "should_escalate": True},
    {"id": "oos-flight", "question": "Book me a flight to Paris", "should_escalate": True},
]

CONFIG = """
[paths]
sources_dir = "sources"
data_dir = "kb_data"
eval_set = "eval_set.jsonl"

[schedule]
update_time = "{update_time}"
timezone = "UTC"
retry_delays_minutes = [15, 30, 60]
poll_seconds = 30

[maintenance_window]
start = "{window_start}"
end = "{window_end}"

[health]
watch_minutes = 5
check_interval_seconds = 30
max_latency_ms = 3000
failure_threshold = 1
max_error_rate = 0.2
min_samples = 5

[quality]
min_accuracy = 0.8
min_grounding = 0.9
min_entries = 3

[security]
max_failed_auth = 5
failed_auth_window_minutes = 10
lockout_minutes = 15
{security_extra}

[monitoring]
min_samples = 5
alert_cooldown_minutes = 30
"""

# 01:30 UTC on a Tuesday: inside the default 01:00-05:00 window.
INSIDE_WINDOW = datetime(2026, 11, 24, 1, 30, tzinfo=timezone.utc)
OUTSIDE_WINDOW = datetime(2026, 11, 24, 14, 0, tzinfo=timezone.utc)


@dataclass
class Project:
    root: Path
    cfg: PipelineConfig
    clock: FakeClock
    faults: FaultInjector = field(default_factory=FaultInjector)
    _pipeline: KnowledgeBasePipeline | None = None

    @property
    def sources(self) -> Path:
        return self.cfg.paths.sources_dir

    @property
    def data(self) -> Path:
        return self.cfg.paths.data_dir

    @property
    def pipeline(self) -> KnowledgeBasePipeline:
        if self._pipeline is None:
            self._pipeline = KnowledgeBasePipeline(self.cfg, clock=self.clock, faults=self.faults)
        return self._pipeline

    def control(self, client_id: str = "test") -> ControlPlane:
        plane = ControlPlane(self.cfg, clock=self.clock, faults=self.faults, client_id=client_id)
        plane.pipeline = self.pipeline
        return plane

    def service(self, **kwargs) -> ChatService:
        return ChatService(self.cfg, clock=self.clock, faults=self.faults, **kwargs)

    def write(self, name: str, text: str, encoding: str = "utf-8") -> Path:
        path = self.sources / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode(encoding))
        return path

    def write_bytes(self, name: str, data: bytes) -> Path:
        path = self.sources / name
        path.write_bytes(data)
        return path

    def active(self) -> str | None:
        return self.pipeline.store.active_version()

    def active_questions(self) -> set[str]:
        return {e.question for e in self.pipeline.store.load_entries(self.active())}


def make_project(root: Path, *, now: datetime = INSIDE_WINDOW, update_time: str = "02:00",
                 window: tuple[str, str] = ("01:00", "05:00"), security_extra: str = "",
                 faq: str | None = FAQ_CSV) -> Project:
    (root / "sources").mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(CONFIG.format(update_time=update_time, window_start=window[0],
                                                    window_end=window[1], security_extra=security_extra),
                                      encoding="utf-8")
    with open(root / "eval_set.jsonl", "w", encoding="utf-8") as handle:
        for item in EVAL_ITEMS:
            handle.write(json.dumps(item) + "\n")
    if faq is not None:
        (root / "sources" / "faq.csv").write_text(faq, encoding="utf-8")
    cfg = load_config(root / "config.toml", root=root)
    return Project(root=root, cfg=cfg, clock=FakeClock(now))


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return make_project(tmp_path)


@pytest.fixture
def live_project(tmp_path: Path) -> Project:
    """A project whose first version (v000001) is already active and stable."""
    proj = make_project(tmp_path)
    report = proj.pipeline.run_update()
    assert report.status == "activated", report.to_dict()
    proj.clock.advance(minutes=6)
    proj.pipeline.tick()
    assert proj.pipeline.status()["health_watch"] is None
    return proj
