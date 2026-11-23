"""The knowledge-base update pipeline.

One update run:

1. scan the sources folder and compare each file's hash with the base
   version's manifest - only new or modified files are parsed;
2. skip exact copies of documents that are already ingested;
3. validate each changed file; invalid or poisoned files are moved to
   ``data_dir/quarantine`` with a ``.reason.json`` (if a previously good file
   becomes invalid, its last good entries stay live);
4. mask personal data, merge with the carried-over entries and remove
   duplicate / near-duplicate entries;
5. build a new immutable version and run the quality tests against it and
   against the active version; reject it if accuracy or grounding drops;
6. activate an approved version only inside the maintenance window, then
   watch its health for ``health.watch_minutes`` and roll back
   automatically if a health check fails.

Failed runs (outages, locked files, write errors) are retried after 15, 30
and 60 minutes. ``tick()`` drives the schedule, retries, pending activation
and health watch, and is what the daemon calls every ``poll_seconds``.
"""

from __future__ import annotations

import copy
import re
import secrets
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator

from .clock import Clock, SystemClock
from .config import PipelineConfig
from .dedupe import dedupe_entries, fingerprint
from .documents import KBEntry, SourceFile, build_entries, load_document, make_entry_id, scan_sources, sha256_bytes
from .engine import AnswerEngine, CircuitBreaker, Generator
from .errors import LockHeldError, PipelineError, StoreError, TransientError, ValidationError
from .faults import FaultInjector
from .fsutil import FileLock, atomic_write_json, read_json
from .health import HealthChecker, HealthResult
from .index import TfidfIndex
from .monitoring import AlertManager, JsonlLog, Metrics
from .quality import EvalItem, EvalResult, GateDecision, QualityGate, evaluate, load_eval_set
from .scheduling import MaintenanceWindow, RetryPolicy, UpdateSchedule
from .security import InjectionDetector, PIIMasker
from .store import VersionStore

DEFAULT_STATE: dict[str, Any] = {
    "pending": None,          # approved version waiting for the maintenance window
    "retry": None,            # {"failures", "next_at", "last_error", ...}
    "health_watch": None,     # post-activation watch
    "schedule": {"last_handled": None},
    "last_update": None,
}


def _iso(value: datetime) -> str:
    return value.isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass
class UpdateReport:
    update_id: str
    trigger: str
    actor: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"   # no_changes | pending_activation | activated | rejected | failed
    attempt: int = 1
    base_version: str | None = None
    candidate_version: str | None = None
    processed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    retained: list[str] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    duplicate_documents: list[dict[str, Any]] = field(default_factory=list)
    duplicate_entries: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    quarantined: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pii_masked: dict[str, int] = field(default_factory=dict)
    entry_count: int = 0
    gate: dict[str, Any] | None = None
    activation: dict[str, Any] | None = None
    error: str | None = None
    next_retry_at: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id, "status": self.status, "trigger": self.trigger,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "candidate_version": self.candidate_version, "base_version": self.base_version,
            "processed": len(self.processed), "unchanged": len(self.unchanged), "deleted": len(self.deleted),
            "quarantined": len(self.quarantined), "duplicate_documents": len(self.duplicate_documents),
            "duplicate_entries": len(self.duplicate_entries), "error": self.error,
            "next_retry_at": self.next_retry_at, "message": self.message,
        }


class KnowledgeBasePipeline:
    def __init__(self, cfg: PipelineConfig, *, clock: Clock | None = None, faults: FaultInjector | None = None,
                 generator: Generator | None = None,
                 notifiers: list[Callable[[dict[str, Any]], None]] | None = None) -> None:
        self.cfg = cfg
        self.clock = clock or SystemClock(cfg.tz)
        self.faults = faults if faults is not None else FaultInjector.from_env()
        self.generator = generator
        self.data_dir = cfg.paths.data_dir
        self.logs_dir = self.data_dir / "logs"
        self.quarantine_dir = self.data_dir / "quarantine"
        for folder in (self.data_dir, self.logs_dir, self.quarantine_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.store = VersionStore(self.data_dir / "versions", faults=self.faults)
        self.state_path = self.data_dir / "state.json"
        self.registry_path = self.data_dir / "registry.json"
        self.lock = FileLock(self.data_dir / ".update.lock")
        self._lock_depth = 0
        self.events = JsonlLog(self.logs_dir / "events.jsonl", self.clock)
        self.audit = JsonlLog(self.logs_dir / "audit.jsonl", self.clock)
        self.reports = JsonlLog(self.logs_dir / "updates.jsonl", self.clock)
        self.metrics = Metrics(self.events, self.clock)
        self.alerts = AlertManager(self.logs_dir / "alerts.jsonl", self.clock, cfg.monitoring, self.metrics,
                                   notifiers, min_confidence=cfg.retrieval.min_confidence)
        self.masker = PIIMasker(cfg.security.pii_allowlist)
        self.detector = InjectionDetector(cfg.security.injection_block_threshold,
                                          cfg.security.injection_suspicious_threshold)
        self.gate = QualityGate(cfg.quality, self.masker, self.detector)
        self.window = MaintenanceWindow(cfg.window.start, cfg.window.end, cfg.window.days)
        self.schedule = UpdateSchedule(cfg.schedule.update_time, cfg.schedule.days)
        self.retry_policy = RetryPolicy(cfg.schedule.retry_delays_minutes)
        self.breaker = CircuitBreaker()
        self._eval_items: list[EvalItem] | None = None

    # ================================================================== helpers
    @property
    def eval_items(self) -> list[EvalItem]:
        if self._eval_items is None:
            self._eval_items = load_eval_set(self.cfg.paths.eval_set)
        return self._eval_items

    @property
    def health(self) -> HealthChecker:
        return HealthChecker(self.cfg.health, engine_loader=self.load_engine, metrics=self.metrics,
                             faults=self.faults)

    def make_engine(self, label: str, entries: list[KBEntry], index: TfidfIndex) -> AnswerEngine:
        return AnswerEngine(label, entries, index, self.cfg.retrieval, generator=self.generator,
                            breaker=self.breaker, grounding_threshold=self.cfg.quality.grounding_threshold,
                            faults=self.faults)

    def load_engine(self, version: str) -> AnswerEngine:
        """Load a stored version (checksums verified on every load)."""
        return self.make_engine(version, self.store.load_entries(version), self.store.load_index(version))

    def _state(self) -> dict[str, Any]:
        state = copy.deepcopy(DEFAULT_STATE)
        state.update(read_json(self.state_path, {}) or {})
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_path, state)

    def _registry(self) -> dict[str, dict[str, Any]]:
        return (read_json(self.registry_path, {}) or {}).get("paths", {})

    def _save_registry(self, registry: dict[str, dict[str, Any]]) -> None:
        atomic_write_json(self.registry_path, {"paths": registry})

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Inter-process lock; re-entrant inside this process."""
        if self._lock_depth:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        self.lock.acquire()
        self._lock_depth = 1
        try:
            yield
        finally:
            self._lock_depth = 0
            self.lock.release()

    def _base_version(self, state: dict[str, Any]) -> str | None:
        pending = state.get("pending")
        if pending and self.store.exists(pending["version"]):
            return pending["version"]
        return self.store.active_version()

    # ================================================================== update
    def run_update(self, *, trigger: str = "manual", actor: str = "system") -> UpdateReport:
        now = self.clock.now()
        state = self._state()
        retry = state.get("retry")
        report = UpdateReport(update_id=f"u{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}", trigger=trigger,
                              actor=actor, started_at=_iso(now),
                              attempt=(retry["failures"] + 1) if retry else 1)
        try:
            with self._locked():
                self._update(report)
        except ValidationError as exc:  # should not escape _update, but never crash the scheduler
            self._on_failure(report, exc)
        except (TransientError, StoreError, OSError, PipelineError) as exc:
            self._on_failure(report, exc)
        except Exception as exc:  # unexpected bug: report and retry like any failure
            self._on_failure(report, exc)
        else:
            self._on_success(report)
        report.finished_at = _iso(self.clock.now())
        self.reports.append("update_report", **report.to_dict())
        self.metrics.record("update", status=report.status, update_id=report.update_id, trigger=trigger,
                            processed=len(report.processed), quarantined=len(report.quarantined),
                            duplicate_documents=len(report.duplicate_documents),
                            duplicate_entries=len([d for d in report.duplicate_entries
                                                   if d["dropped_source"] in report.processed]),
                            candidate=report.candidate_version, error=report.error)
        state = self._state()
        state["last_update"] = report.summary()
        self._save_state(state)
        return report

    def _update(self, report: UpdateReport) -> None:
        cfg = self.cfg
        state = self._state()
        base = self._base_version(state)
        report.base_version = base
        base_manifest: dict[str, Any] = self.store.load_manifest(base)["sources"] if base else {}
        base_source_entries: dict[str, list[KBEntry]] = {}
        if base:
            for entry in self.store.load_source_entries(base):
                base_source_entries.setdefault(entry.source, []).append(entry)
        registry = self._registry()
        registry_before = copy.deepcopy(registry)

        self.faults.check("source_read")
        sources = scan_sources(cfg.paths.sources_dir)
        present = {s.path for s in sources}

        # ---- pass 1: hash every file (fast path: unchanged size + mtime)
        hashes: dict[str, str] = {}
        raw_cache: dict[str, bytes] = {}
        for src in sources:
            prev = base_manifest.get(src.path)
            if prev and prev.get("size") == src.size and prev.get("mtime_ns") == src.mtime_ns:
                hashes[src.path] = prev["sha256"]
                continue
            try:
                raw = src.abs_path.read_bytes()
            except OSError as exc:
                raise TransientError(f"cannot read {src.path}: {exc}") from exc
            raw_cache[src.path] = raw
            hashes[src.path] = sha256_bytes(raw)

        new_manifest: dict[str, Any] = {}
        carried: list[KBEntry] = []
        fresh: list[KBEntry] = []
        source_rank: dict[str, float] = {}
        known_content: dict[str, str] = {}   # content hash -> path already in the KB

        # ---- pass 2: unchanged files carry their entries forward
        changed: list[SourceFile] = []
        for src in sources:
            prev = base_manifest.get(src.path)
            if prev and prev.get("sha256") == hashes[src.path]:
                report.unchanged.append(src.path)
                new_manifest[src.path] = {**prev, "size": src.size, "mtime_ns": src.mtime_ns}
                carried.extend(base_source_entries.get(src.path, []))
                known_content[prev["content_hash"]] = src.path
                source_rank[src.path] = float(src.mtime_ns)
            else:
                changed.append(src)

        # ---- pass 3: skip files already known to be quarantined / duplicates
        to_process: list[SourceFile] = []
        for src in changed:
            reg = registry.get(src.path)
            if reg and reg.get("sha256") == hashes[src.path]:
                if reg.get("status") == "quarantined":
                    report.skipped.append({"path": src.path, "status": "still_quarantined", "reason": reg.get("reason")})
                    self._retain(src.path, base_manifest, base_source_entries, new_manifest, carried, report)
                    continue
                if reg.get("status") == "duplicate" and reg.get("duplicate_of") in known_content.values():
                    report.skipped.append({"path": src.path, "status": "duplicate", "duplicate_of": reg["duplicate_of"]})
                    continue
            to_process.append(src)

        # ---- pass 4: deleted / quarantined-away files
        for path, prev in base_manifest.items():
            if path in present:
                continue
            reg = registry.get(path)
            if reg and reg.get("status") == "quarantined":
                self._retain(path, base_manifest, base_source_entries, new_manifest, carried, report)
            else:
                report.deleted.append(path)

        # ---- pass 5: process new / modified files
        for src in to_process:
            report.processed.append(src.path)
            raw = raw_cache.get(src.path)
            if raw is None:
                try:
                    raw = src.abs_path.read_bytes()
                except OSError as exc:
                    raise TransientError(f"cannot read {src.path}: {exc}") from exc
            try:
                doc = load_document(src.path, raw, cfg.ingestion)
                duplicate_of = known_content.get(doc.content_hash)
                if duplicate_of:
                    report.duplicate_documents.append({"path": src.path, "duplicate_of": duplicate_of})
                    registry[src.path] = {"sha256": hashes[src.path], "status": "duplicate",
                                          "duplicate_of": duplicate_of, "at": _iso(self.clock.now())}
                    continue
                flagged = []
                for row in doc.rows:
                    verdict = self.detector.scan(f"{row.question}\n{row.answer}")
                    if verdict.blocked:
                        flagged.append(f"row {row.row}: {', '.join(verdict.matches)} (score {verdict.score})")
                    elif verdict.verdict == "suspicious":
                        report.warnings.append(f"{src.path} row {row.row}: suspicious text ({', '.join(verdict.matches)})")
                if flagged:
                    raise ValidationError("prompt_injection", flagged)
                entries = build_entries(src.path, doc.rows, cfg.ingestion)
            except ValidationError as exc:
                self._quarantine(src, raw, hashes[src.path], exc, report, registry)
                self._retain(src.path, base_manifest, base_source_entries, new_manifest, carried, report)
                continue

            masked_entries = []
            for entry in entries:
                mq, ma = self.masker.mask(entry.question), self.masker.mask(entry.answer)
                for label, count in {**mq.counts}.items():
                    report.pii_masked[label] = report.pii_masked.get(label, 0) + count
                for label, count in ma.counts.items():
                    report.pii_masked[label] = report.pii_masked.get(label, 0) + count
                masked_entries.append(KBEntry(make_entry_id(src.path, mq.text, ma.text), mq.text, ma.text,
                                              entry.source, entry.row))
            fresh.extend(masked_entries)
            known_content[doc.content_hash] = src.path
            source_rank[src.path] = float(src.mtime_ns)
            registry.pop(src.path, None)
            new_manifest[src.path] = {"sha256": hashes[src.path], "content_hash": doc.content_hash,
                                      "size": src.size, "mtime_ns": src.mtime_ns, "status": "active",
                                      "encoding": doc.encoding, "entries": len(masked_entries)}
            report.warnings.extend(f"{src.path}: {w}" for w in doc.warnings)

        # registry entries for duplicates of deleted originals are stale
        for path in list(registry):
            if registry[path].get("status") == "duplicate" and path not in present:
                registry.pop(path)

        if registry != registry_before:
            self._save_registry(registry)

        if not report.processed and not report.deleted:
            report.status = "no_changes"
            report.message = "no new, modified or deleted documents"
            return

        # ---- merge + de-duplicate
        all_entries = carried + fresh
        deduped = dedupe_entries(all_entries, source_rank=source_rank,
                                 threshold=cfg.ingestion.near_duplicate_threshold)
        entries = deduped.entries
        report.duplicate_entries = [d.to_dict() for d in deduped.duplicates]
        report.conflicts = [c.to_dict() for c in deduped.conflicts]
        report.entry_count = len(entries)
        content_fp = fingerprint(entries)

        base_meta = self.store.load_meta(base) if base else None
        if base_meta and base_meta.get("fingerprint") == content_fp and set(new_manifest) == set(base_manifest):
            report.status = "no_changes"
            report.message = "changed files produced no new knowledge (quarantined, duplicate or identical)"
            return

        rolled_back_match = None
        for version in self.store.list_versions():
            try:
                meta = self.store.load_meta(version)
            except StoreError:
                continue
            if meta.get("status") == "rolled_back" and meta.get("fingerprint") == content_fp:
                rolled_back_match = version

        # ---- build + quality gate
        self.faults.check("index_build")
        index = TfidfIndex.build(entries)
        candidate = evaluate(self.make_engine("candidate", entries, index), self.eval_items,
                             grounding_threshold=cfg.quality.grounding_threshold)
        active = self.store.active_version()
        baseline = self._evaluate_version(active, report)
        decision = self.gate.check(candidate, baseline, entries, rolled_back_match=rolled_back_match)
        report.gate = decision.to_dict()

        meta = {
            "status": "approved" if decision.approved else "rejected",
            "created_at": _iso(self.clock.now()),
            "parent": base,
            "baseline": active,
            "update_id": report.update_id,
            "trigger": report.trigger,
            "fingerprint": content_fp,
            "eval": decision.candidate,
            "gate_reasons": decision.reasons,
            "sources": len(new_manifest),
        }
        version = self.store.create_version(entries, index, {"sources": new_manifest}, meta,
                                            source_entries=all_entries)
        report.candidate_version = version

        if not decision.approved:
            report.status = "rejected"
            report.message = "; ".join(decision.reasons)
            self.alerts.raise_alert("update_rejected", "warning",
                                    f"update {version} rejected: {report.message}", dedupe=False, version=version)
            self._prune(state)
            return

        state = self._state()
        old_pending = state.get("pending")
        if old_pending and old_pending.get("version") != version and self.store.exists(old_pending["version"]):
            self.store.set_status(old_pending["version"], "superseded", superseded_by=version)
        state["pending"] = {"version": version, "approved_at": _iso(self.clock.now()), "baseline_version": active,
                            "update_id": report.update_id}
        self._save_state(state)
        report.status = "pending_activation"
        report.message = f"approved; waiting for the maintenance window ({self.window.describe()})"

        outcome = self.activate_pending(actor=f"pipeline:{report.trigger}")
        report.activation = outcome
        if outcome.get("rolled_back"):
            report.status = "rolled_back"
            report.message = f"{version} was activated but failed its first health check and was rolled back"
        elif outcome.get("activated"):
            report.status = "activated"
            report.message = f"activated {version}; health watch until {outcome['watch_until']}"
        self._prune(self._state())

    def _retain(self, path: str, base_manifest: dict[str, Any], base_entries: dict[str, list[KBEntry]],
                new_manifest: dict[str, Any], carried: list[KBEntry], report: UpdateReport) -> None:
        """Keep the last good entries of a source that is now quarantined."""
        prev = base_manifest.get(path)
        if prev and prev.get("status") == "active" and path not in new_manifest:
            new_manifest[path] = prev
            carried.extend(base_entries.get(path, []))
            report.retained.append(path)

    def _quarantine(self, src: SourceFile, raw: bytes, sha: str, exc: ValidationError,
                    report: UpdateReport, registry: dict[str, Any]) -> None:
        now = self.clock.now()
        safe = re.sub(r"[^\w.-]+", "_", src.path)
        dest = self.quarantine_dir / f"{now:%Y%m%dT%H%M%S}__{safe}"
        moved = False
        try:
            shutil.move(str(src.abs_path), str(dest))
            moved = True
        except OSError:
            try:  # cannot move (locked / read-only source): keep a copy and remember its hash
                dest.write_bytes(raw)
            except OSError:
                pass
        record = {"original_path": src.path, "reason": exc.reason, "details": exc.details[:20], "sha256": sha,
                  "quarantined_at": _iso(now), "moved_out_of_sources": moved, "update_id": report.update_id}
        try:
            atomic_write_json(dest.with_name(dest.name + ".reason.json"), record)
        except TransientError:
            pass
        registry[src.path] = {"sha256": sha, "status": "quarantined", "reason": exc.reason,
                              "details": exc.details[:5], "at": _iso(now), "stored_as": dest.name, "moved": moved}
        report.quarantined.append({"path": src.path, "reason": exc.reason, "details": exc.details[:5],
                                   "stored_as": dest.name})
        self.metrics.record("quarantine", path=src.path, reason=exc.reason)
        self.alerts.raise_alert("file_quarantined", "warning", f"{src.path} quarantined: {exc.reason}",
                                dedupe=False, path=src.path, reason=exc.reason)

    def _evaluate_version(self, version: str | None, report: UpdateReport | None = None) -> EvalResult | None:
        if not version:
            return None
        try:
            return evaluate(self.load_engine(version), self.eval_items,
                            grounding_threshold=self.cfg.quality.grounding_threshold)
        except (StoreError, TransientError) as exc:
            if report is not None:
                report.warnings.append(f"active version {version} could not be evaluated ({exc}); "
                                       f"using absolute thresholds only")
            return None

    def _on_failure(self, report: UpdateReport, exc: BaseException) -> None:
        now = self.clock.now()
        report.status = "failed"
        report.error = f"{type(exc).__name__}: {exc}"
        state = self._state()
        retry = state.get("retry") or {"failures": 0, "first_failed_at": _iso(now)}
        retry["failures"] = int(retry.get("failures", 0)) + 1
        retry["last_error"] = report.error
        retry["last_failed_at"] = _iso(now)
        next_at = self.retry_policy.next_retry_at(retry["failures"], now)
        if next_at is not None:
            retry["next_at"] = _iso(next_at)
            state["retry"] = retry
            report.next_retry_at = _iso(next_at)
            delay = self.retry_policy.delays[retry["failures"] - 1]
            report.message = f"attempt {retry['failures']} failed; retrying in {delay:g} minutes"
            self.metrics.record("update_retry_scheduled", failures=retry["failures"], next_at=_iso(next_at),
                                error=report.error)
            self.alerts.raise_alert("update_failed", "warning",
                                    f"update failed ({report.error}); retry {retry['failures']} of "
                                    f"{self.retry_policy.max_retries} at {next_at:%H:%M}", dedupe=False)
        else:
            state["retry"] = None
            report.message = (f"failed {retry['failures']} times; giving up until the next scheduled run "
                              f"({self.schedule.next_occurrence(now):%Y-%m-%d %H:%M})")
            self.metrics.record("update_failed_permanently", failures=retry["failures"], error=report.error)
            self.alerts.raise_alert("update_failed_permanently", "critical",
                                    f"update failed after {retry['failures']} attempts: {report.error}", dedupe=False)
        self._save_state(state)

    def _on_success(self, report: UpdateReport) -> None:
        state = self._state()
        if state.get("retry"):
            state["retry"] = None
            self._save_state(state)

    def _prune(self, state: dict[str, Any]) -> None:
        protect = {p for p in [(state.get("pending") or {}).get("version"),
                               (state.get("health_watch") or {}).get("previous")] if p}
        target = self.store.rollback_target()
        if target:
            protect.add(target)
        self.store.prune(self.cfg.versions.keep, protect)

    # ================================================================== activation
    def activate_pending(self, *, actor: str = "system") -> dict[str, Any]:
        with self._locked():
            now = self.clock.now()
            # finish (or act on) a previous health watch first, so updates run
            # from the CLI without the daemon are not blocked by a stale watch
            self.health_tick()
            state = self._state()
            pending = state.get("pending")
            if not pending:
                return {"activated": False, "reason": "no approved update is waiting"}
            if state.get("health_watch"):
                return {"activated": False,
                        "reason": f"health watch for {state['health_watch']['version']} is still running"}
            if not self.window.contains(now):
                return {"activated": False,
                        "reason": f"outside the maintenance window {self.window.describe()}; "
                                  f"next opens {self.window.next_open(now):%Y-%m-%d %H:%M}"}
            version = pending["version"]
            active = self.store.active_version()
            if pending.get("baseline_version") != active:
                decision = self._regate(version, active)
                if not decision.approved:
                    self.store.set_status(version, "rejected", gate_reasons=decision.reasons)
                    state["pending"] = None
                    self._save_state(state)
                    self.alerts.raise_alert("update_rejected", "warning",
                                            f"{version} failed re-validation: {'; '.join(decision.reasons)}", dedupe=False)
                    return {"activated": False, "reason": "re-validation failed: " + "; ".join(decision.reasons)}
            previous = self.store.activate(version, at=now, actor=actor, reason="approved update")
            watch_until = now + timedelta(minutes=self.cfg.health.watch_minutes)
            state["pending"] = None
            state["health_watch"] = {"version": version, "previous": previous, "activated_at": _iso(now),
                                     "until": _iso(watch_until), "next_check_at": _iso(now),
                                     "failures": 0, "checks": 0}
            self._save_state(state)
            self.metrics.record("activation", version=version, previous=previous, actor=actor)
            first = self.health_tick()  # immediate first check
            return {"activated": True, "rolled_back": bool(first and first.get("event") == "rollback"),
                    "version": version, "previous": previous, "watch_until": _iso(watch_until),
                    "first_health_check": first}

    def _regate(self, version: str, active: str | None) -> GateDecision:
        entries = self.store.load_entries(version)
        candidate = evaluate(self.make_engine(version, entries, self.store.load_index(version)), self.eval_items,
                             grounding_threshold=self.cfg.quality.grounding_threshold)
        return self.gate.check(candidate, self._evaluate_version(active), entries)

    # ================================================================== health
    def health_tick(self) -> dict[str, Any] | None:
        """Run a due health check for a freshly activated version; roll back
        on failure inside the watch period."""
        state = self._state()
        watch = state.get("health_watch")
        if not watch:
            return None
        now = self.clock.now()
        if self.store.active_version() != watch["version"]:
            state["health_watch"] = None
            self._save_state(state)
            return {"event": "watch_cancelled", "version": watch["version"], "reason": "active version changed"}
        until = _dt(watch["until"])
        activated_at = _dt(watch["activated_at"])
        within_watch = now <= until
        outcome: dict[str, Any] | None = None
        if now >= _dt(watch["next_check_at"]):
            result = self.health.check(watch["version"], since=activated_at)
            watch["checks"] += 1
            watch["last_check_at"] = _iso(now)
            self.metrics.record("health_check", ok=result.ok, version=watch["version"], error=result.error,
                                within_watch=within_watch)
            outcome = {"event": "health_check", "ok": result.ok, "version": watch["version"], "error": result.error}
            if result.ok:
                watch["failures"] = 0
            else:
                watch["failures"] += 1
                if within_watch and watch["failures"] >= self.cfg.health.failure_threshold:
                    return self._auto_rollback(state, watch, result)
                if not within_watch:
                    self.alerts.raise_alert("health_degraded", "critical",
                                            f"{watch['version']} failed a health check after the watch period: "
                                            f"{result.error}", version=watch["version"])
            watch["next_check_at"] = _iso(now + timedelta(seconds=self.cfg.health.check_interval_seconds))
        if now >= until:
            state["health_watch"] = None
            self.store.update_meta(watch["version"], health="stable" if watch["failures"] == 0 else "degraded",
                                   watch_finished_at=_iso(now), health_checks=watch["checks"])
            self._save_state(state)
            self.metrics.record("activation_stable", version=watch["version"], checks=watch["checks"])
            return outcome or {"event": "watch_finished", "version": watch["version"]}
        state["health_watch"] = watch
        self._save_state(state)
        return outcome

    def _auto_rollback(self, state: dict[str, Any], watch: dict[str, Any], result: HealthResult) -> dict[str, Any]:
        now = self.clock.now()
        reason = f"health check failed {watch['failures']} time(s) within {self.cfg.health.watch_minutes:g} " \
                 f"minutes of activation: {result.error}"
        state["health_watch"] = None
        try:
            from_version, to_version = self.store.rollback(at=now, actor="system:health", reason=reason)
        except StoreError as exc:
            self._save_state(state)
            self.store.update_meta(watch["version"], health="failed")
            self.alerts.raise_alert("rollback_impossible", "critical",
                                    f"{watch['version']} is unhealthy but cannot be rolled back: {exc}", dedupe=False)
            return {"event": "rollback_failed", "version": watch["version"], "error": str(exc)}
        self._save_state(state)
        self.metrics.record("rollback", automatic=True, from_version=from_version, to_version=to_version,
                            reason=reason)
        self.alerts.raise_alert("automatic_rollback", "critical",
                                f"rolled back {from_version} -> {to_version}: {reason}", dedupe=False)
        return {"event": "rollback", "automatic": True, "from": from_version, "to": to_version, "reason": reason}

    def rollback(self, *, actor: str, to: str | None = None, reason: str = "manual rollback") -> dict[str, Any]:
        with self._locked():
            now = self.clock.now()
            from_version, to_version = self.store.rollback(at=now, actor=actor, reason=reason, to=to)
            state = self._state()
            state["health_watch"] = None
            self._save_state(state)
            self.metrics.record("rollback", automatic=False, from_version=from_version, to_version=to_version,
                                reason=reason, actor=actor)
            self.alerts.raise_alert("manual_rollback", "warning",
                                    f"{actor} rolled back {from_version} -> {to_version}", dedupe=False)
            return {"from": from_version, "to": to_version, "at": _iso(now)}

    # ================================================================== scheduler
    def tick(self) -> dict[str, Any]:
        """One scheduler step: health watch, pending activation, retries,
        the daily schedule, then alert evaluation."""
        now = self.clock.now()
        try:
            with self._locked():
                actions: list[Any] = []
                health = self.health_tick()
                if health:
                    actions.append(health)
                state = self._state()
                if state.get("pending") and not state.get("health_watch") and self.window.contains(now):
                    actions.append({"event": "activation", **self.activate_pending(actor="scheduler")})
                state = self._state()
                retry = state.get("retry")
                if retry and now >= _dt(retry["next_at"]):
                    report = self.run_update(trigger="retry")
                    actions.append({"event": "update", **report.summary()})
                    state = self._state()
                occurrence = self.schedule.latest_occurrence(now)
                last = _dt(state["schedule"].get("last_handled"))
                due = occurrence is not None and (
                    (last is None and now - occurrence <= timedelta(hours=1)) or (last is not None and occurrence > last))
                if occurrence is not None and (due or last is None):
                    state["schedule"]["last_handled"] = _iso(occurrence)
                    self._save_state(state)
                if due:
                    if state.get("retry"):
                        actions.append({"event": "schedule_skipped", "reason": "a retry is already scheduled"})
                    else:
                        report = self.run_update(trigger="schedule")
                        actions.append({"event": "update", **report.summary()})
                raised = self.alerts.evaluate()
                if raised:
                    actions.append({"event": "alerts", "raised": raised})
                return {"now": _iso(now), "actions": actions}
        except LockHeldError as exc:
            return {"now": _iso(now), "actions": [{"event": "busy", "reason": str(exc)}]}

    # ================================================================== reporting
    def status(self) -> dict[str, Any]:
        now = self.clock.now()
        state = self._state()
        active = self.store.active()
        active_meta = None
        if active and active.get("version"):
            try:
                meta = self.store.load_meta(active["version"])
                active_meta = {k: meta.get(k) for k in ("status", "entry_count", "created_at", "health")}
                active_meta["accuracy"] = (meta.get("eval") or {}).get("accuracy")
                active_meta["grounding"] = (meta.get("eval") or {}).get("grounding")
            except StoreError as exc:
                active_meta = {"error": str(exc)}
        registry = self._registry()
        return {
            "now": _iso(now),
            "active": {**(active or {}), **({"meta": active_meta} if active_meta else {})} if active else None,
            "pending": state.get("pending"),
            "health_watch": state.get("health_watch"),
            "retry": state.get("retry"),
            "schedule": {"update_time": f"{self.cfg.schedule.update_time:%H:%M}",
                         "next_run": _iso(self.schedule.next_occurrence(now)),
                         "last_handled": state["schedule"].get("last_handled")},
            "maintenance_window": {"window": self.window.describe(), "open_now": self.window.contains(now),
                                   "next_open": _iso(self.window.next_open(now))},
            "last_update": state.get("last_update"),
            "versions": len(self.store.list_versions()),
            "quarantined_files": sum(1 for r in registry.values() if r.get("status") == "quarantined"),
        }

    def versions(self) -> list[dict[str, Any]]:
        out = []
        active = self.store.active_version()
        for version in reversed(self.store.list_versions()):
            try:
                meta = self.store.load_meta(version)
            except StoreError:
                out.append({"version": version, "status": "unreadable"})
                continue
            out.append({"version": version, "status": meta.get("status"), "active": version == active,
                        "created_at": meta.get("created_at"), "entries": meta.get("entry_count"),
                        "accuracy": (meta.get("eval") or {}).get("accuracy"),
                        "grounding": (meta.get("eval") or {}).get("grounding"),
                        "parent": meta.get("parent"), "gate_reasons": meta.get("gate_reasons") or [],
                        "health": meta.get("health")})
        return out

    def quarantine_list(self) -> list[dict[str, Any]]:
        return [{"path": path, **record} for path, record in sorted(self._registry().items())
                if record.get("status") == "quarantined"]

    def release_quarantine(self, path: str, *, actor: str) -> dict[str, Any]:
        """Put a quarantined file back into the sources folder (after it was
        fixed by hand) so the next update processes it again."""
        with self._locked():
            registry = self._registry()
            record = registry.get(path)
            if not record or record.get("status") != "quarantined":
                raise PipelineError(f"{path} is not quarantined")
            stored = self.quarantine_dir / record["stored_as"]
            target = self.cfg.paths.sources_dir / path
            if record.get("moved") and stored.is_file():
                if target.exists():
                    raise PipelineError(f"{target} already exists; remove it first")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(stored), str(target))
            registry.pop(path)
            self._save_registry(registry)
            self.audit.append("quarantine_release", path=path, actor=actor)
            return {"released": path, "restored_to": str(target)}
