"""Unauthorised access is denied, audited, rate-limited and alerted."""

import json

import pytest
from conftest import make_project

from kb_pipeline.cli import main
from kb_pipeline.errors import AccessDenied, AuthLockedOut


@pytest.fixture
def plane(live_project):
    return live_project.control(client_id="attacker-1")


@pytest.fixture
def tokens(plane):
    admin = plane.bootstrap("root")
    return {
        "admin": admin,
        "operator": plane.create_user(admin, "ops", "operator"),
        "viewer": plane.create_user(admin, "dash", "viewer"),
        "chat": plane.create_user(admin, "widget", "chat"),
    }


def test_bootstrap_only_once(plane, tokens):
    with pytest.raises(AccessDenied):
        plane.bootstrap("second-admin")


def test_tokens_are_stored_hashed(live_project, tokens):
    stored = (live_project.data / "access.json").read_text(encoding="utf-8")
    for token in tokens.values():
        assert token not in stored
    assert "token_sha256" in stored


@pytest.mark.parametrize("token", [None, "", "kb_not-a-real-token"])
def test_missing_or_invalid_token_is_denied(plane, tokens, token):
    with pytest.raises(AccessDenied):
        plane.update(token)
    with pytest.raises(AccessDenied):
        plane.status(token)


@pytest.mark.parametrize("role,allowed,denied", [
    ("viewer", ["status", "versions", "metrics", "alerts", "quarantine", "escalations"],
     ["update", "activate", "rollback", "users"]),
    ("operator", ["status", "update", "activate", "tick"], ["rollback", "users", "release"]),
    ("chat", [], ["status", "metrics", "update", "rollback"]),
    ("admin", ["status", "update", "users", "rollback_attempt"], []),
])
def test_role_permissions(plane, tokens, role, allowed, denied):
    token = tokens[role]
    calls = {
        "status": lambda: plane.status(token),
        "versions": lambda: plane.versions(token),
        "metrics": lambda: plane.metrics(token),
        "alerts": lambda: plane.alerts(token),
        "quarantine": lambda: plane.quarantine(token),
        "escalations": lambda: plane.escalations(token),
        "update": lambda: plane.update(token),
        "activate": lambda: plane.activate(token),
        "tick": lambda: plane.tick(token),
        "rollback": lambda: plane.rollback(token),
        "users": lambda: plane.users(token),
        "release": lambda: plane.release(token, "nothing.csv"),
    }
    for name in allowed:
        if name == "rollback_attempt":  # authorised, but there is nothing to roll back to
            with pytest.raises(Exception) as info:
                plane.rollback(token)
            assert not isinstance(info.value, AccessDenied)
            continue
        calls[name]()
    for name in denied:
        with pytest.raises(AccessDenied):
            calls[name]()


def test_denials_are_audited_and_counted(live_project, plane, tokens):
    with pytest.raises(AccessDenied):
        plane.rollback(tokens["viewer"])
    audit = [json.loads(line) for line in
             (live_project.data / "logs" / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    denied = [a for a in audit if a.get("kind") == "access" and a.get("allowed") is False]
    assert denied[-1]["permission"] == "kb:rollback" and denied[-1]["user"] == "dash"
    assert tokens["viewer"] not in json.dumps(audit)
    summary = live_project.pipeline.metrics.summary()
    assert summary["security"]["auth_denied"] >= 1


def test_brute_force_locks_out_client(live_project, plane, tokens):
    for _ in range(5):
        with pytest.raises(AccessDenied):
            plane.status("kb_guess")
    with pytest.raises(AuthLockedOut):
        plane.status(tokens["admin"])               # even a valid token is refused while locked
    other_client = live_project.control(client_id="office-laptop")
    assert other_client.status(tokens["admin"])     # lockout is per client
    assert "auth_lockout" in [a["kind"] for a in live_project.pipeline.alerts.recent()]
    live_project.clock.advance(minutes=16)
    assert plane.status(tokens["admin"])            # lock expires


def test_repeated_denials_raise_unauthorised_access_alert(live_project, plane, tokens):
    for client in range(6):
        with pytest.raises(AccessDenied):
            live_project.control(client_id=f"bot-{client}").update(tokens["chat"])
    raised = live_project.pipeline.alerts.evaluate()
    assert "unauthorised_access" in raised


def test_revoked_token_is_denied(plane, tokens):
    assert plane.revoke_user(tokens["admin"], "ops")
    with pytest.raises(AccessDenied, match="revoked"):
        plane.update(tokens["operator"])


def test_anonymous_chat_allowed_and_configurable(tmp_path, live_project):
    assert live_project.service().ask("Do you offer a refund?").version == "v000001"
    locked = make_project(tmp_path / "private", security_extra="allow_anonymous_chat = false")
    with pytest.raises(AccessDenied):
        locked.service().ask("Do you offer a refund?")


def test_cli_returns_exit_code_2_when_denied(live_project, capsys):
    config = str(live_project.root / "config.toml")
    assert main(["--config", config, "update"]) == 2
    assert "access denied" in capsys.readouterr().err
    assert main(["--config", config, "--token", "kb_wrong", "rollback"]) == 2
