# Knowledge-base pipeline: design and runbook

The pipeline keeps the chatbot's knowledge base up to date without ever putting
a broken, poisoned or lower-quality version in front of learners.

```
dataset/ ──scan──► change detection ──► validate ──► injection scan ──► mask PII ──► merge + dedupe
 (sources)        (hash vs manifest)       │              │                              │
                                           └──► quarantine/ (file + .reason.json) ◄──────┘
                                                                                         ▼
     kb_data/versions/v00000N  ◄── build index ◄── quality gate (eval set vs active) ◄── candidate
                │ approved                                │ rejected → kept for audit, alert
                ▼
   pending ──(maintenance window)──► activate ──► health watch 5 min ──► stable
                                                    │ check fails
                                                    ▼
                                          automatic rollback + alert
```

All code lives in `src/kb_pipeline/`; everything is plain Python standard library.

| Module | Responsibility |
|---|---|
| `documents.py` | find, decode (UTF-8 / Windows-1252), parse and validate CSV, JSON, Markdown and text files |
| `dedupe.py` | exact / near-duplicate / conflicting entries, content fingerprints |
| `security.py` | prompt-injection detector, sensitive-data masker |
| `index.py` | TF-IDF retriever (one immutable index per version) |
| `engine.py` | answer or escalate; optional LLM generator with grounding check and circuit breaker |
| `quality.py` | evaluation set, accuracy and grounding metrics, quality gate |
| `store.py` | versioned storage, checksums, activation history, rollback, pruning |
| `scheduling.py` | daily schedule, 15/30/60-minute retry policy, maintenance window |
| `health.py` | post-activation health checks |
| `pipeline.py` | orchestrates updates, activation, health watch and the scheduler `tick()` |
| `access.py` | roles, hashed tokens, lockout, audit log |
| `monitoring.py` | event log, metric summaries, threshold alerts |
| `service.py` | runtime chat service used by the Streamlit app |
| `control.py` / `cli.py` | access-controlled operations, `kbctl.py` |

## Requirements and how they are met

| Requirement | Implementation | Tests |
|---|---|---|
| Process only new or modified documents | Each version stores a manifest (path → SHA-256, size, mtime). Unchanged files are not parsed; their entries are carried forward. Deleted files' entries are removed. Nothing changed → no new version. | `test_ingestion.py` |
| Detect duplicates | Whole-document copies (same normalised content, even with other encodings or line endings) are skipped and remembered. Entries are de-duplicated exactly and by near-match (question **and** answer word overlap ≥ 0.9). Same question with a different answer is a *conflict*: the newer file wins and it is reported. | `test_duplicates.py` |
| Quarantine invalid files | Empty, binary, unsupported, oversized, malformed (missing columns, ragged rows, empty answers, bad JSON) and prompt-injected files are moved to `kb_data/quarantine/` with a `.reason.json`. If a previously good file turns bad, its last good entries stay live. `kbctl release` returns a fixed file. | `test_quarantine.py` |
| Versions and rollback | Immutable `vNNNNNN` folders, written atomically, every file checksummed and verified on load. Full activation history. `kbctl rollback [--to]` at any time; rejected and rolled-back versions can never be re-activated. Old versions are pruned (`versions.keep`) except the active one and rollback targets. | `test_versions_and_quality.py` |
| Quality tests before activation | Every candidate is evaluated on `config/eval_set.jsonl` (in-scope questions with required facts + out-of-scope questions that must be escalated). Rejected if accuracy or grounding is below the minimum **or lower than the active version**, if any entry still contains personal data or injection text, or if its content equals a version that was rolled back. | `test_versions_and_quality.py` |
| Configurable schedule | `[schedule] update_time` (+ optional weekdays, timezone). Missed runs are caught up once. | `test_scheduling.py` |
| Retry after 15, 30, 60 minutes | Outages and errors (unreadable sources, storage failures, a held lock) schedule retries at +15, +30 and +60 minutes after each consecutive failure; a 4th failure gives up until the next scheduled run and raises a critical alert. Quality rejections are not retried (they would fail again). | `test_scheduling.py` |
| Activate only in the maintenance window | Approved versions wait as *pending*; the scheduler activates them when the window opens (windows may cross midnight). A newer approval supersedes the pending one. Manual `activate` outside the window is refused. | `test_scheduling.py` |
| Automatic rollback within 5 minutes | Right after activation and then every 30 s for 5 minutes, health checks verify file integrity, answer smoke questions within the latency limit and watch the live error rate. A failure inside the watch rolls back to the previous good version and raises a critical alert. After 5 minutes the version is marked stable; later failures alert only. | `test_health_rollback.py` |
| Access control | Roles chat / viewer / operator / admin. Random 256-bit tokens, stored only as SHA-256 hashes. 5 failures in 10 minutes lock that client out for 15 minutes. Every decision goes to `logs/audit.jsonl`; repeated denials raise an `unauthorised_access` alert. Learners can chat anonymously (`security.allow_anonymous_chat`). | `test_access_control.py` |
| Prompt-injection protection | Weighted rules for instruction overrides, system-prompt extraction, role/persona switches, jailbreak phrases, chat-template tokens, exfiltration and hidden HTML, checked on normalised, de-leeted and letter-collapsed text so zero-width characters, `1gn0re` and `i g n o r e` don't slip through. Applied to every document row (file quarantined) and every question (refused and logged). Retrieved text is fenced as data in the LLM prompt. | `test_security.py` |
| Sensitive-data masking | E-mail, phone (Indian and international), card numbers (Luhn-checked), Aadhaar, PAN, SSN, API keys/tokens, JWTs, passwords, IP addresses and secret URL parameters. Applied to documents before indexing, to questions before search/logging, and to answers. Numeric IDs inside links are left alone; public contacts can be allow-listed. | `test_security.py` |
| Monitoring | `logs/events.jsonl` (no raw questions) → `kbctl metrics`: latency p50/p95/p99, error rate, escalation rate, average and low confidence, blocked injections, updates by status, retries, rollbacks, health checks, denied access. Threshold alerts with cool-down in `logs/alerts.jsonl`; escalated questions (masked) in `logs/escalations.jsonl`. Dashboard: Streamlit **Monitoring** page. | `test_monitoring_and_outages.py` |
| Service outages | Missing sources or storage failures → update fails safely and is retried; the active version keeps serving. Chat keeps answering from the last loaded version if storage becomes unreadable (`degraded`), escalates politely if nothing is loaded, and an LLM outage falls back to the stored answer behind a circuit breaker. | `test_monitoring_and_outages.py` |

## Rehearsing the evaluation scenarios

Every scenario can be reproduced from the CLI without waiting for real time:
`--now` fixes the clock and `--faults` injects outages
(`source_read`, `index_build`, `store_write`, `store_read`, `health_check`, `generator`;
`name=N` fails N times, `name=always` keeps failing). The same can be set with the
`KB_FAULTS` environment variable.

Try them on a copy of the project folder, because the drills add files to `dataset/`.

```bash
export KB_TOKEN=<admin token>          # PowerShell: $env:KB_TOKEN = "<admin token>"

# First build inside the window -> "activated"
python kbctl.py --now 2026-11-24T01:30 update

# Duplicate document: the same FAQ under another name -> "duplicate_documents", no new version
cp dataset/dataset.csv dataset/dataset_copy.csv
python kbctl.py --now 2026-11-24T01:40 update

# Failed update: index outage -> exit code 4, "next_retry_at" 15 minutes later
printf 'question,answer\nDo you have a mobile app?,"Yes, on Android."\n' > dataset/app.csv
python kbctl.py --faults index_build=always --now 2026-11-24T02:00 update

# Health failure right after activation -> "rolled_back" + automatic_rollback alert
python kbctl.py --faults health_check=always --now 2026-11-24T02:05 update

# Outside the maintenance window -> "pending_activation"; manual activate is refused
printf 'question,answer\nWhere are you located?,"We are fully online."\n' > dataset/loc.csv
python kbctl.py --now 2026-11-24T14:00 update
python kbctl.py --now 2026-11-24T14:00 activate

# Simulated night 23:50 -> 02:30 in 30-second ticks: the pending update activates at 01:00,
# the 02:00 scheduled run finds no changes, each activation gets its 5-minute health watch
python kbctl.py --now 2026-11-24T23:50 daemon --iterations 320

# Unauthorised access -> exit code 2, audited, 5 failures lock the client out
python kbctl.py --token wrong status

python kbctl.py versions
python kbctl.py alerts
```

Exit codes: `0` ok, `1` error, `2` access denied, `3` update rejected, `4` update failed.

## Operating notes

* **First run.** `python kbctl.py bootstrap` creates the first admin token (only possible
  while no users exist). Store it in a password manager; it is never shown again.
* **Daemon.** Run `python kbctl.py daemon` as a service (Task Scheduler on Windows,
  systemd on Linux). It needs an operator token in `KB_TOKEN`. Only one update can run at
  a time (`kb_data/.update.lock`; a lock older than an hour is treated as stale).
* **Adding knowledge.** Drop or edit files in `dataset/`. The next scheduled run picks up
  only what changed. Check `kbctl quarantine` and `kbctl alerts` after a run.
* **Updating the evaluation set.** Add questions to `config/eval_set.jsonl` whenever you add
  important FAQs, so future updates cannot silently break them.
* **Plugging in an LLM.** Pass any callable `generator(system=..., question=..., contexts=...)`
  to `ChatService` / `KnowledgeBasePipeline`. Its answers are used only when at least 80 % of
  their words are supported by the retrieved entries; otherwise the stored answer is shown.

## Known limitations

* Retrieval is TF-IDF with a small synonym map, so heavily reworded questions can miss
  (the shipped eval set scores 97 %; "I have no coding experience" does not find the
  "never done programming" entry). Those questions are escalated to staff rather than answered
  wrongly. An embedding retriever can replace `TfidfIndex` behind the same interface.
* The injection detector is rule-based: it stops the common and obfuscated patterns tested here,
  not every possible attack. The grounding check and fenced context are the second line of defence.
* Metrics are JSON-lines files on local disk. For several app servers, ship them to a central
  log store.
