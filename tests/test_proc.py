"""Unit tests for proc.run_capture's env= passthrough.

Regression coverage for the bug where claude -p dispatches always inherited
JARVIS's own ANTHROPIC_API_KEY, which disables claude.ai connectors in the
Claude Code CLI. run_capture had no env= parameter at all, so there was no
way for a caller to strip it before spawning the child.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from proc import run_capture, env_without


@pytest.mark.asyncio
async def test_env_none_inherits_parent_environment(monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_VAR", "inherited")
    _, stdout, _ = await run_capture(
        sys.executable, "-c", "import os; print(os.environ.get('JARVIS_TEST_VAR', ''))"
    )
    assert stdout.decode().strip() == "inherited"


@pytest.mark.asyncio
async def test_explicit_env_replaces_parent_environment(monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_VAR", "should_not_appear")
    _, stdout, _ = await run_capture(
        sys.executable, "-c", "import os; print(os.environ.get('JARVIS_TEST_VAR', 'absent'))",
        env={"PATH": "/usr/bin:/bin"},
    )
    assert stdout.decode().strip() == "absent"


@pytest.mark.asyncio
async def test_explicit_env_can_omit_a_single_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("OTHER_VAR", "kept")
    _, stdout, _ = await run_capture(
        sys.executable, "-c",
        "import os; print(os.environ.get('ANTHROPIC_API_KEY', 'absent'), os.environ.get('OTHER_VAR', 'absent'))",
        env=env_without("ANTHROPIC_API_KEY"),
    )
    assert stdout.decode().strip() == "absent kept"


def test_env_without_omits_only_the_named_keys(monkeypatch):
    monkeypatch.setenv("KEEP_ME", "1")
    monkeypatch.setenv("DROP_ME", "1")
    filtered = env_without("DROP_ME")
    assert "DROP_ME" not in filtered
    assert filtered.get("KEEP_ME") == "1"


def test_env_without_recomputes_on_every_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" not in env_without("ANTHROPIC_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-set-later")
    # Still stripped — proves this isn't a cached snapshot from import time.
    assert "ANTHROPIC_API_KEY" not in env_without("ANTHROPIC_API_KEY")
