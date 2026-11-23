"""Duplicate detection for knowledge-base entries.

* exact duplicate   - same normalised question and answer
* near duplicate    - question AND answer word-overlap (Jaccard) >= threshold
* conflict          - same normalised question but a different answer; the
                      entry from the most recently modified source wins

Whole-document duplicates (the same file saved twice under different names)
are caught earlier by content hash in the pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .documents import KBEntry
from .textutil import jaccard, normalize


@dataclass
class DuplicateRecord:
    kind: str           # "exact" | "near"
    kept_id: str
    dropped_id: str
    kept_source: str
    dropped_source: str
    similarity: float
    question: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConflictRecord:
    kept_id: str
    dropped_id: str
    kept_source: str
    dropped_source: str
    question: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DedupeResult:
    entries: list[KBEntry]
    duplicates: list[DuplicateRecord] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)


def dedupe_entries(entries: list[KBEntry], *, source_rank: dict[str, float] | None = None,
                   threshold: float = 0.9) -> DedupeResult:
    """Entries are considered in order: for duplicates the earlier entry wins,
    so pass existing entries before newly ingested ones."""
    rank = source_rank or {}
    kept: list[KBEntry] = []
    signatures: list[tuple[set[str], set[str]]] = []
    by_question: dict[str, int] = {}
    result = DedupeResult(entries=kept)

    for entry in entries:
        nq, na = normalize(entry.question), normalize(entry.answer)
        index = by_question.get(nq)
        if index is not None:
            existing = kept[index]
            if normalize(existing.answer) == na:
                result.duplicates.append(DuplicateRecord(
                    "exact", existing.id, entry.id, existing.source, entry.source, 1.0, entry.question))
                continue
            newer = entry.source != existing.source and rank.get(entry.source, 0) > rank.get(existing.source, 0)
            if newer:
                kept[index] = entry
                signatures[index] = (set(nq.split()), set(na.split()))
                result.conflicts.append(ConflictRecord(entry.id, existing.id, entry.source, existing.source, entry.question))
            else:
                result.conflicts.append(ConflictRecord(existing.id, entry.id, existing.source, entry.source, entry.question))
            continue

        q_tokens, a_tokens = set(nq.split()), set(na.split())
        near: tuple[int, float] | None = None
        for i, (kq, ka) in enumerate(signatures):
            q_sim = jaccard(q_tokens, kq)
            if q_sim < threshold:
                continue
            a_sim = jaccard(a_tokens, ka)
            if a_sim >= threshold:
                near = (i, min(q_sim, a_sim))
                break
        if near is not None:
            existing = kept[near[0]]
            result.duplicates.append(DuplicateRecord(
                "near", existing.id, entry.id, existing.source, entry.source, round(near[1], 3), entry.question))
            continue

        by_question[nq] = len(kept)
        kept.append(entry)
        signatures.append((q_tokens, a_tokens))
    return result


def fingerprint(entries: list[KBEntry]) -> str:
    """Order-independent hash of the knowledge content (used to spot a
    candidate identical to the active or a rolled-back version)."""
    import hashlib

    parts = sorted(f"{normalize(e.question)}\x1f{normalize(e.answer)}" for e in entries)
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()
