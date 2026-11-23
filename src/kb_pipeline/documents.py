"""Source-document discovery, decoding, parsing and validation.

Supported formats:

* ``.csv``  - FAQ sheets with a question column (question/prompt/query) and an
  answer column (answer/response/reply).
* ``.json`` - a list of ``{"question": ..., "answer": ...}`` objects (or
  ``{"entries": [...]}``).
* ``.md`` / ``.txt`` - ``Q:``/``A:`` pairs, Markdown ``#`` sections, or plain
  paragraphs.

Anything that cannot be parsed cleanly raises :class:`ValidationError`, which
the pipeline turns into a quarantine record.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import IngestionConfig
from .errors import TransientError, ValidationError
from .textutil import clean_text, normalize

QUESTION_COLUMNS = ("question", "questions", "prompt", "query", "q")
ANSWER_COLUMNS = ("answer", "answers", "response", "reply", "a")
IGNORED_NAMES = {".gitkeep", "thumbs.db", "desktop.ini", ".ds_store", "readme.md"}


@dataclass(frozen=True)
class KBEntry:
    id: str
    question: str
    answer: str
    source: str
    row: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "KBEntry":
        return cls(id=data["id"], question=data["question"], answer=data["answer"],
                   source=data["source"], row=int(data.get("row", 0)))


def make_entry_id(source: str, question: str, answer: str) -> str:
    digest = hashlib.sha256(
        f"{source}\x1f{normalize(question)}\x1f{normalize(answer)}".encode("utf-8")
    ).hexdigest()
    return "e_" + digest[:16]


@dataclass
class SourceFile:
    path: str          # POSIX path relative to the sources directory
    abs_path: Path
    size: int
    mtime_ns: int


@dataclass
class RawRow:
    row: int
    question: str
    answer: str


@dataclass
class ParsedDocument:
    rows: list[RawRow]
    encoding: str
    content_hash: str
    warnings: list[str] = field(default_factory=list)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan_sources(sources_dir: Path) -> list[SourceFile]:
    """List candidate files. A missing directory is treated as an outage
    (for example an unmounted share) so the update is retried later."""
    if not sources_dir.is_dir():
        raise TransientError(f"sources directory is unavailable: {sources_dir}")
    found: list[SourceFile] = []
    try:
        for path in sorted(sources_dir.rglob("*")):
            rel = path.relative_to(sources_dir)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if not path.is_file():
                continue
            name = path.name.lower()
            if name in IGNORED_NAMES or name.startswith("~$") or name.endswith((".tmp", ".partial")):
                continue
            stat = path.stat()
            found.append(SourceFile(rel.as_posix(), path, stat.st_size, stat.st_mtime_ns))
    except OSError as exc:
        raise TransientError(f"cannot list sources directory {sources_dir}: {exc}") from exc
    return found


def decode_bytes(raw: bytes, encodings: tuple[str, ...]) -> tuple[str, str]:
    if b"\x00" in raw:
        raise ValidationError("binary_content", "file contains NUL bytes (not a text document)")
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise ValidationError("undecodable", f"could not decode as any of: {', '.join(encodings)}")
    control = sum(1 for ch in text if (ord(ch) < 32 and ch not in "\n\r\t") or 0x7F <= ord(ch) < 0xA0)
    if control > max(3, len(text) // 100):
        raise ValidationError("binary_content", f"{control} control characters found")
    return text, encoding


def content_hash(text: str) -> str:
    """Hash of the decoded text with line endings and trailing spaces
    normalised, so the same document saved with another encoding or line
    ending style is still recognised as a duplicate."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    return sha256_bytes("\n".join(line.rstrip() for line in lines).encode("utf-8"))


# ----------------------------------------------------------------- parsers

def _pick_column(header: list[str], aliases: tuple[str, ...]) -> int | None:
    lowered = [h.strip().lower().lstrip("﻿") for h in header]
    for alias in aliases:
        if alias in lowered:
            return lowered.index(alias)
    return None


def parse_csv(text: str) -> list[RawRow]:
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader, None)
        if not header or not any(h.strip() for h in header):
            raise ValidationError("missing_header", "CSV has no header row")
        q_idx = _pick_column(header, QUESTION_COLUMNS)
        a_idx = _pick_column(header, ANSWER_COLUMNS)
        if q_idx is None or a_idx is None:
            raise ValidationError(
                "missing_columns",
                f"need a question column ({'/'.join(QUESTION_COLUMNS)}) and an answer column "
                f"({'/'.join(ANSWER_COLUMNS)}); found {header}",
            )
        rows: list[RawRow] = []
        problems: list[str] = []
        for record in reader:
            line = reader.line_num
            if not record or all(not cell.strip() for cell in record):
                continue
            if len(record) != len(header):
                problems.append(f"line {line}: expected {len(header)} columns, found {len(record)}")
                continue
            rows.append(RawRow(line, record[q_idx], record[a_idx]))
    except csv.Error as exc:
        raise ValidationError("malformed_csv", str(exc)) from exc
    if problems:
        raise ValidationError("malformed_rows", problems)
    return rows


def parse_json(text: str) -> list[RawRow]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError("malformed_json", f"line {exc.lineno}: {exc.msg}") from exc
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        data = data["entries"]
    if not isinstance(data, list):
        raise ValidationError("invalid_structure", "expected a list of {question, answer} objects")
    rows: list[RawRow] = []
    problems: list[str] = []
    for number, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            problems.append(f"item {number}: not an object")
            continue
        lowered = {str(k).lower(): v for k, v in item.items()}
        question = next((lowered[k] for k in QUESTION_COLUMNS if k in lowered), None)
        answer = next((lowered[k] for k in ANSWER_COLUMNS if k in lowered), None)
        if not isinstance(question, str) or not isinstance(answer, str):
            problems.append(f"item {number}: needs string 'question' and 'answer' fields")
            continue
        rows.append(RawRow(number, question, answer))
    if problems:
        raise ValidationError("malformed_rows", problems)
    return rows


_Q_RE = re.compile(r"^\s*(?:q|question)\s*[:.)\-]\s*(.*)$", re.IGNORECASE)
_A_RE = re.compile(r"^\s*(?:a|answer)\s*[:.)\-]\s*(.*)$", re.IGNORECASE)
_H_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def parse_text(text: str) -> list[RawRow]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    rows: list[RawRow] = []

    if any(_Q_RE.match(line) for line in lines) and any(_A_RE.match(line) for line in lines):
        question: str | None = None
        q_line = 0
        answer: list[str] | None = None
        problems: list[str] = []

        def flush() -> None:
            if question is None:
                return
            if answer is None:
                problems.append(f"line {q_line}: question without an answer")
            else:
                rows.append(RawRow(q_line, question, "\n".join(answer).strip()))

        for number, line in enumerate(lines, start=1):
            q_match, a_match = _Q_RE.match(line), _A_RE.match(line)
            if q_match:
                flush()
                question, q_line, answer = q_match.group(1).strip(), number, None
            elif a_match and question is not None and answer is None:
                answer = [a_match.group(1)]
            elif answer is not None:
                answer.append(line)
            elif question is not None and line.strip():
                question = f"{question} {line.strip()}"
        flush()
        if problems:
            raise ValidationError("malformed_rows", problems)
        return rows

    if any(_H_RE.match(line) for line in lines):
        heading: str | None = None
        start = 0
        body: list[str] = []
        for number, line in enumerate(lines + ["# __end__"], start=1):
            match = _H_RE.match(line)
            if match:
                if heading is not None and "\n".join(body).strip():
                    rows.append(RawRow(start, heading, "\n".join(body).strip()))
                heading, start, body = match.group(1), number, []
            else:
                body.append(line)
        return rows

    paragraph: list[str] = []
    start = 1
    for number, line in enumerate(lines + [""], start=1):
        if line.strip():
            if not paragraph:
                start = number
            paragraph.append(line.strip())
        elif paragraph:
            first = paragraph[0]
            title = first if len(first) <= 150 else first[:147] + "..."
            rows.append(RawRow(start, title, " ".join(paragraph)))
            paragraph = []
    return rows


PARSERS = {".csv": parse_csv, ".json": parse_json, ".md": parse_text, ".txt": parse_text}


def load_document(path: str, raw: bytes, cfg: IngestionConfig) -> ParsedDocument:
    """Validate and parse one file. Raises ValidationError on any problem."""
    ext = Path(path).suffix.lower()
    if ext not in cfg.allowed_extensions or ext not in PARSERS:
        raise ValidationError("unsupported_type", f"'{ext or '(no extension)'}' files are not accepted")
    if len(raw) > cfg.max_file_bytes:
        raise ValidationError("file_too_large", f"{len(raw)} bytes exceeds the {cfg.max_file_bytes} byte limit")
    if not raw.strip():
        raise ValidationError("empty_file", "file is empty")
    text, encoding = decode_bytes(raw, cfg.encodings)
    rows = PARSERS[ext](text)
    if not rows:
        raise ValidationError("no_entries", "no question/answer pairs found")
    warnings: list[str] = []
    if encoding not in ("utf-8", "utf-8-sig"):
        warnings.append(f"decoded as {encoding} (not UTF-8)")
    return ParsedDocument(rows=rows, encoding=encoding, content_hash=content_hash(text), warnings=warnings)


def build_entries(source: str, rows: list[RawRow], cfg: IngestionConfig) -> list[KBEntry]:
    """Clean rows into entries, enforcing length limits."""
    entries: list[KBEntry] = []
    problems: list[str] = []
    for row in rows:
        question, answer = clean_text(row.question), clean_text(row.answer)
        if len(question) < 2:
            problems.append(f"row {row.row}: empty question")
        elif len(question) > cfg.max_question_chars:
            problems.append(f"row {row.row}: question longer than {cfg.max_question_chars} characters")
        elif not answer:
            problems.append(f"row {row.row}: empty answer")
        elif len(answer) > cfg.max_answer_chars:
            problems.append(f"row {row.row}: answer longer than {cfg.max_answer_chars} characters")
        else:
            entries.append(KBEntry(make_entry_id(source, question, answer), question, answer, source, row.row))
    if problems:
        raise ValidationError("invalid_entries", problems)
    return entries
