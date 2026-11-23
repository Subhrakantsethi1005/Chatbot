# Customer Service Chatbot with a Production Knowledge-Base Pipeline

A Q&A chatbot for an e-learning company (Nullclass) that sells data courses and
virtual internships. Learners ask questions in a Streamlit app and get answers
from the company's FAQ sheet within milliseconds; questions the bot is not
sure about are escalated to the support team.

The knowledge base is maintained by a production pipeline that only processes
new or modified documents, detects duplicates, quarantines invalid files,
versions every update with rollback, runs quality tests before activation,
activates only inside a maintenance window, rolls back automatically on failed
health checks, and monitors latency, failures, confidence and escalations.
See **[docs/PIPELINE.md](docs/PIPELINE.md)** for the full design and runbook.

## Quick start

```bash
git clone https://github.com/Subhrakantsethi1005/Chatbot.git
cd Chatbot
pip install -r requirements.txt

# 1. create your admin token (shown once - keep it safe)
python kbctl.py bootstrap --name admin

# 2. build the knowledge base from dataset/ (pass the token or set KB_TOKEN)
python kbctl.py --token <TOKEN> update

# 3. start the chatbot
streamlit run src/main.py
```

Updates are only *activated* inside the maintenance window (default 01:00-05:00).
Outside it, `update` builds and approves the new version and it goes live at the
next window. To try everything right now, simulate the time:

```bash
python kbctl.py --token <TOKEN> --now 2026-11-24T01:30 update
```

For production, keep the scheduler running (daily update at 02:00 by default):

```bash
python kbctl.py --token <TOKEN> daemon
```

## Everyday commands

| Command | Role | What it does |
|---|---|---|
| `python kbctl.py status` | viewer | active version, pending update, window, retries |
| `python kbctl.py versions` | viewer | all versions with accuracy / grounding / status |
| `python kbctl.py metrics --window 60` | viewer | latency p50/p95/p99, error, escalation and confidence stats |
| `python kbctl.py alerts` | viewer | recent alerts |
| `python kbctl.py escalations` | viewer | questions handed to staff |
| `python kbctl.py quarantine` | viewer | quarantined files and why |
| `python kbctl.py update` | operator | run an update now |
| `python kbctl.py activate` | operator | activate the approved update (window only) |
| `python kbctl.py rollback [--to v000003]` | admin | roll back |
| `python kbctl.py user-add --name dash --role viewer` | admin | issue a token |
| `python kbctl.py ask "Do you offer EMI?"` | anyone | ask from the terminal |

The monitoring dashboard is the **Monitoring** page of the Streamlit app (needs a viewer token).

## Project structure

```
config/pipeline.toml     schedule, window, thresholds, security settings
config/eval_set.jsonl    quality tests run before every activation
dataset/                 source documents (.csv / .json / .md / .txt)
kb_data/                 runtime data: versions, quarantine, logs (git-ignored)
kbctl.py                 operations CLI
src/main.py              Streamlit chatbot
src/pages/1_Monitoring.py  monitoring dashboard
src/kb_pipeline/         the pipeline package
tests/                   pytest suite covering every requirement
src/langchain_helper.py  original Google PaLM version (no longer used; PaLM is shut down)
```

## Tests

```bash
python -m pytest
```

## Sample questions

- Do you provide any virtual internship?
- Should I learn Power BI or Tableau?
- I have a Mac. Can I use Power BI on it?
- How do I enable Power Pivot?
- Do you have a JavaScript course? *(escalated - not in the FAQ)*
