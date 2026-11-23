"""Pipeline configuration, loaded from ``config/pipeline.toml``.

Unknown keys are rejected so that a typo (for example ``retry_delay`` instead
of ``retry_delays_minutes``) fails loudly instead of silently using a default.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.toml"

WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
ALL_DAYS = frozenset(range(7))


def parse_hhmm(value: Any, key: str = "time") -> time:
    try:
        hours, minutes = str(value).strip().split(":")
        return time(int(hours), int(minutes))
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{key}: invalid time {value!r}, expected HH:MM (24-hour)") from exc


def parse_days(value: Any, key: str = "days") -> frozenset[int]:
    if not value:
        return ALL_DAYS
    try:
        return frozenset(WEEKDAYS[str(day).strip().lower()[:3]] for day in value)
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"{key}: expected weekday names like ['mon', 'tue'], got {value!r}") from exc


def resolve_timezone(name: str) -> tzinfo:
    if not name or name.lower() == "local":
        local = datetime.now().astimezone().tzinfo
        return local or timezone.utc
    if name.upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(f"schedule.timezone: unknown timezone {name!r}") from exc


@dataclass
class PathsConfig:
    sources_dir: Path
    data_dir: Path
    eval_set: Path


@dataclass
class ScheduleConfig:
    update_time: time = time(2, 0)
    days: frozenset[int] = ALL_DAYS
    timezone: str = "local"
    retry_delays_minutes: tuple[float, ...] = (15, 30, 60)
    poll_seconds: float = 30


@dataclass
class WindowConfig:
    start: time = time(1, 0)
    end: time = time(5, 0)
    days: frozenset[int] = ALL_DAYS


@dataclass
class HealthConfig:
    watch_minutes: float = 5
    check_interval_seconds: float = 30
    max_latency_ms: float = 500
    failure_threshold: int = 1
    max_error_rate: float = 0.2
    min_samples: int = 5
    smoke_queries: tuple[str, ...] = ()


@dataclass
class QualityConfig:
    min_accuracy: float = 0.8
    min_grounding: float = 0.9
    max_regression: float = 0.0
    min_entries: int = 1
    grounding_threshold: float = 0.8


@dataclass
class IngestionConfig:
    allowed_extensions: tuple[str, ...] = (".csv", ".json", ".md", ".txt")
    max_file_bytes: int = 5_000_000
    encodings: tuple[str, ...] = ("utf-8-sig", "cp1252")
    near_duplicate_threshold: float = 0.9
    max_question_chars: int = 500
    max_answer_chars: int = 8000


@dataclass
class RetrievalConfig:
    min_confidence: float = 0.25
    top_k: int = 3


@dataclass
class SecurityConfig:
    allow_anonymous_chat: bool = True
    max_failed_auth: int = 5
    failed_auth_window_minutes: float = 10
    lockout_minutes: float = 15
    injection_block_threshold: float = 0.7
    injection_suspicious_threshold: float = 0.4
    max_query_chars: int = 1000
    pii_allowlist: tuple[str, ...] = ()


@dataclass
class VersionsConfig:
    keep: int = 10


@dataclass
class MonitoringConfig:
    window_minutes: float = 60
    min_samples: int = 10
    alert_p95_latency_ms: float = 1500
    alert_error_rate: float = 0.05
    alert_escalation_rate: float = 0.35
    alert_min_avg_confidence: float = 0.35
    alert_auth_failures: int = 5
    alert_cooldown_minutes: float = 30


@dataclass
class PipelineConfig:
    root: Path
    paths: PathsConfig
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    versions: VersionsConfig = field(default_factory=VersionsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    source_file: Path | None = None

    @property
    def tz(self) -> tzinfo:
        return resolve_timezone(self.schedule.timezone)


# --------------------------------------------------------------------------- loading

def _section(raw: dict, name: str, spec: dict[str, Callable[[Any], Any]]) -> dict[str, Any]:
    data = raw.get(name, {}) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] must be a table")
    unknown = sorted(set(data) - set(spec))
    if unknown:
        raise ConfigError(f"[{name}] has unknown key(s): {', '.join(unknown)}")
    out: dict[str, Any] = {}
    for key, convert in spec.items():
        if key in data:
            try:
                out[key] = convert(data[key])
            except ConfigError:
                raise
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"[{name}] {key}: invalid value {data[key]!r}") from exc
    return out


def _positive(convert: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def inner(value: Any) -> Any:
        result = convert(value)
        if result <= 0:
            raise ValueError("must be positive")
        return result
    return inner


def _fraction(value: Any) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("must be between 0 and 1")
    return result


def _str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    return tuple(str(v) for v in value)


def _delays(value: Any) -> tuple[float, ...]:
    delays = tuple(float(v) for v in value)
    if any(d <= 0 for d in delays):
        raise ValueError("retry delays must be positive")
    return delays


def _extensions(value: Any) -> tuple[str, ...]:
    return tuple(("." + str(v).lower().lstrip(".")) for v in value)


def load_config(path: str | Path | None = None, *, root: str | Path | None = None) -> PipelineConfig:
    """Load the TOML config. Relative paths are resolved against ``root``
    (default: the project root that contains ``config/``)."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc

    known_sections = {"paths", "schedule", "maintenance_window", "health", "quality", "ingestion",
                      "retrieval", "security", "versions", "monitoring"}
    unknown = sorted(set(raw) - known_sections)
    if unknown:
        raise ConfigError(f"unknown config section(s): {', '.join(unknown)}")

    base = Path(root) if root else (config_path.resolve().parent.parent
                                     if config_path.resolve().parent.name == "config"
                                     else config_path.resolve().parent)

    def rel(value: Any) -> Path:
        p = Path(str(value))
        return p if p.is_absolute() else (base / p)

    paths = _section(raw, "paths", {"sources_dir": rel, "data_dir": rel, "eval_set": rel})
    paths_cfg = PathsConfig(
        sources_dir=paths.get("sources_dir", base / "dataset"),
        data_dir=paths.get("data_dir", base / "kb_data"),
        eval_set=paths.get("eval_set", base / "config" / "eval_set.jsonl"),
    )

    schedule = ScheduleConfig(**_section(raw, "schedule", {
        "update_time": lambda v: parse_hhmm(v, "schedule.update_time"),
        "days": lambda v: parse_days(v, "schedule.days"),
        "timezone": str,
        "retry_delays_minutes": _delays,
        "poll_seconds": _positive(float),
    }))
    resolve_timezone(schedule.timezone)  # validate early

    window = WindowConfig(**_section(raw, "maintenance_window", {
        "start": lambda v: parse_hhmm(v, "maintenance_window.start"),
        "end": lambda v: parse_hhmm(v, "maintenance_window.end"),
        "days": lambda v: parse_days(v, "maintenance_window.days"),
    }))
    if window.start == window.end:
        raise ConfigError("maintenance_window: start and end must differ")

    health = HealthConfig(**_section(raw, "health", {
        "watch_minutes": _positive(float),
        "check_interval_seconds": _positive(float),
        "max_latency_ms": _positive(float),
        "failure_threshold": _positive(int),
        "max_error_rate": _fraction,
        "min_samples": int,
        "smoke_queries": _str_tuple,
    }))
    quality = QualityConfig(**_section(raw, "quality", {
        "min_accuracy": _fraction,
        "min_grounding": _fraction,
        "max_regression": _fraction,
        "min_entries": _positive(int),
        "grounding_threshold": _fraction,
    }))
    ingestion = IngestionConfig(**_section(raw, "ingestion", {
        "allowed_extensions": _extensions,
        "max_file_bytes": _positive(int),
        "encodings": _str_tuple,
        "near_duplicate_threshold": _fraction,
        "max_question_chars": _positive(int),
        "max_answer_chars": _positive(int),
    }))
    retrieval = RetrievalConfig(**_section(raw, "retrieval", {
        "min_confidence": _fraction,
        "top_k": _positive(int),
    }))
    security = SecurityConfig(**_section(raw, "security", {
        "allow_anonymous_chat": bool,
        "max_failed_auth": _positive(int),
        "failed_auth_window_minutes": _positive(float),
        "lockout_minutes": _positive(float),
        "injection_block_threshold": _fraction,
        "injection_suspicious_threshold": _fraction,
        "max_query_chars": _positive(int),
        "pii_allowlist": _str_tuple,
    }))
    versions = VersionsConfig(**_section(raw, "versions", {"keep": _positive(int)}))
    monitoring = MonitoringConfig(**_section(raw, "monitoring", {
        "window_minutes": _positive(float),
        "min_samples": int,
        "alert_p95_latency_ms": _positive(float),
        "alert_error_rate": _fraction,
        "alert_escalation_rate": _fraction,
        "alert_min_avg_confidence": _fraction,
        "alert_auth_failures": _positive(int),
        "alert_cooldown_minutes": float,
    }))

    return PipelineConfig(
        root=base, paths=paths_cfg, schedule=schedule, window=window, health=health,
        quality=quality, ingestion=ingestion, retrieval=retrieval, security=security,
        versions=versions, monitoring=monitoring, source_file=config_path,
    )
