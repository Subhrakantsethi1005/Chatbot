"""Command-line interface: ``python kbctl.py <command>``.

Authentication: pass ``--token`` or set the ``KB_TOKEN`` environment
variable. Create the first admin token with ``python kbctl.py bootstrap``.

Simulation: ``--now 2026-11-24T01:30`` runs a command as if it were that
time, and ``--faults index_build=2`` injects outages, so schedules,
maintenance windows, retries and rollbacks can be rehearsed safely, e.g.::

    python kbctl.py --now 2026-11-24T00:50 daemon --iterations 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

from .clock import FakeClock
from .config import load_config
from .control import ControlPlane
from .errors import AccessDenied, ConfigError, PipelineError
from .faults import FaultInjector
from .service import ChatService

EXIT_OK, EXIT_ERROR, EXIT_DENIED, EXIT_REJECTED, EXIT_FAILED = 0, 1, 2, 3, 4


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbctl", description="Operate the chatbot knowledge-base pipeline.")
    parser.add_argument("--config", help="path to pipeline.toml (default: config/pipeline.toml)")
    parser.add_argument("--token", help="access token (default: $KB_TOKEN)")
    parser.add_argument("--now", help="simulate the current time, e.g. 2026-11-24T01:30")
    parser.add_argument("--faults", help="inject outages, e.g. 'index_build=2,health_check=always'")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bootstrap", help="create the first admin token (only when no users exist)")
    p.add_argument("--name", default="admin")
    p = sub.add_parser("user-add", help="issue a token for a new user (admin)")
    p.add_argument("--name", required=True)
    p.add_argument("--role", required=True, choices=["chat", "viewer", "operator", "admin"])
    p = sub.add_parser("user-revoke", help="revoke a user's token (admin)")
    p.add_argument("--name", required=True)
    sub.add_parser("users", help="list users (admin)")
    p = sub.add_parser("audit", help="show the audit log (admin)")
    p.add_argument("--limit", type=int, default=50)

    sub.add_parser("update", help="run an update now (operator)")
    sub.add_parser("tick", help="run one scheduler step (operator)")
    p = sub.add_parser("daemon", help="run the scheduler loop (operator)")
    p.add_argument("--iterations", type=int, help="stop after N ticks (default: run forever)")
    sub.add_parser("activate", help="activate the approved update if the window is open (operator)")
    p = sub.add_parser("rollback", help="roll back to the previous (or a given) version (admin)")
    p.add_argument("--to", help="version id, e.g. v000003")
    p.add_argument("--reason", default="manual rollback")

    sub.add_parser("status", help="pipeline status (viewer)")
    sub.add_parser("versions", help="list versions (viewer)")
    sub.add_parser("health", help="health-check the active version (viewer)")
    p = sub.add_parser("metrics", help="monitoring summary (viewer)")
    p.add_argument("--window", type=float, default=60, help="minutes to summarise (default 60)")
    p = sub.add_parser("alerts", help="recent alerts (viewer)")
    p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("escalations", help="questions escalated to staff (viewer)")
    p.add_argument("--limit", type=int, default=20)
    sub.add_parser("quarantine", help="list quarantined files (viewer)")
    p = sub.add_parser("release", help="return a fixed file from quarantine to the sources (admin)")
    p.add_argument("path", help="original path relative to the sources folder")
    p = sub.add_parser("ask", help="ask the chatbot a question")
    p.add_argument("question")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = args.token or os.environ.get("KB_TOKEN")
    try:
        cfg = load_config(args.config)
        clock = None
        if args.now:
            when = datetime.fromisoformat(args.now)
            clock = FakeClock(when if when.tzinfo else when.replace(tzinfo=cfg.tz))
        faults = FaultInjector.parse(args.faults) if args.faults else None
        if args.command == "ask":
            response = ChatService(cfg, clock=clock, faults=faults).ask(args.question, token=token, client_id="cli")
            _print(response.to_dict())
            return EXIT_OK
        plane = ControlPlane(cfg, clock=clock, faults=faults, client_id="cli")
        command = args.command
        if command == "bootstrap":
            new_token = plane.bootstrap(args.name)
            print(f"Admin token for '{args.name}' (shown once - store it safely):\n{new_token}")
            return EXIT_OK
        if command == "user-add":
            new_token = plane.create_user(token, args.name, args.role)
            print(f"Token for '{args.name}' ({args.role}) - shown once:\n{new_token}")
            return EXIT_OK
        if command == "update":
            report = plane.update(token)
            _print(report)
            return {"rejected": EXIT_REJECTED, "failed": EXIT_FAILED}.get(report["status"], EXIT_OK)
        if command == "daemon":
            plane.daemon(token, iterations=args.iterations,
                         on_tick=lambda r: r["actions"] and _print(r))
            return EXIT_OK
        handlers = {
            "user-revoke": lambda: {"revoked": plane.revoke_user(token, args.name)},
            "users": lambda: plane.users(token),
            "audit": lambda: plane.audit_log(token, args.limit),
            "tick": lambda: plane.tick(token),
            "activate": lambda: plane.activate(token),
            "rollback": lambda: plane.rollback(token, to=args.to, reason=args.reason),
            "status": lambda: plane.status(token),
            "versions": lambda: plane.versions(token),
            "health": lambda: plane.health(token),
            "metrics": lambda: plane.metrics(token, window_minutes=args.window),
            "alerts": lambda: plane.alerts(token, limit=args.limit),
            "escalations": lambda: plane.escalations(token, limit=args.limit),
            "quarantine": lambda: plane.quarantine(token),
            "release": lambda: plane.release(token, args.path),
        }
        _print(handlers[command]())
        return EXIT_OK
    except AccessDenied as exc:
        print(f"access denied: {exc}", file=sys.stderr)
        return EXIT_DENIED
    except (ConfigError, PipelineError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
