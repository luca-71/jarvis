"""Unit tests for work_mode's dispatch environment.

Regression coverage for the incident where every claude -p dispatch failed
to use claude.ai connectors: JARVIS's own ANTHROPIC_API_KEY (loaded from
.env into os.environ at startup) was silently inherited by every dispatched
`claude -p` subprocess, and Claude Code treats a set ANTHROPIC_API_KEY as
taking precedence over the user's claude.ai login.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import work_mode
from proc import env_without


def test_dispatch_strips_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-reach-dispatches")
    assert "ANTHROPIC_API_KEY" not in env_without("ANTHROPIC_API_KEY")


def test_dispatch_env_preserves_other_vars(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("SOME_OTHER_VAR", "kept")
    filtered = env_without("ANTHROPIC_API_KEY")
    assert filtered.get("SOME_OTHER_VAR") == "kept"


def test_dispatch_env_reflects_live_changes_not_a_stale_snapshot(monkeypatch):
    """The filter must be computed per-call — a key set after import (e.g.
    via /api/settings/keys) must still be stripped, not just whatever was
    set at process startup."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" not in env_without("ANTHROPIC_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-set-after-the-fact")
    assert "ANTHROPIC_API_KEY" not in env_without("ANTHROPIC_API_KEY")


def test_work_session_send_uses_env_without(monkeypatch):
    """WorkSession.send() must actually pass the filtered env through to
    run_capture, not just have the helper available."""
    import asyncio

    captured = {}

    async def fake_run_capture(*args, **kwargs):
        captured.update(kwargs)
        return (0, b"ok", b"")

    monkeypatch.setattr(work_mode, "run_capture", fake_run_capture)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setattr(work_mode.shutil, "which", lambda name: "/usr/bin/claude")

    session = work_mode.WorkSession()
    asyncio.run(session.start("/tmp", "test-project"))
    asyncio.run(session.send("hello"))

    assert "env" in captured
    assert "ANTHROPIC_API_KEY" not in captured["env"]
