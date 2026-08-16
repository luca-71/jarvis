"""Unit tests for dispatch_registry — the prompt-aware recent-dispatch cache.

test_status_check_then_different_work_request reproduces the real failure
observed live: a status-check dispatch, followed by a request for genuinely
different work on the same project within the cache window, silently
replayed the status-check's stale answer instead of running the new request.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import dispatch_registry as dr


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "DB_PATH", tmp_path / "test.db")
    yield


def _completed_registry_with(prompt: str, summary: str) -> dr.DispatchRegistry:
    reg = dr.DispatchRegistry()
    did = reg.register("jarvis", "/path/to/jarvis", prompt)
    reg.update_status(did, "completed", response=summary, summary=summary)
    return reg


# -- _prompts_match ----------------------------------------------------------

def test_identical_prompts_match():
    assert dr._prompts_match("dove siamo?", "dove siamo?")


def test_completely_different_prompts_do_not_match():
    a = "Rivedi lo stato attuale del progetto e dimmi cosa è stato implementato finora."
    b = "Fai commit di memory_vault.py e integralo in server.py. Conferma quando fatto."
    assert not dr._prompts_match(a, b)


def test_empty_prompt_never_matches():
    assert not dr._prompts_match("", "dove siamo?")
    assert not dr._prompts_match("dove siamo?", "")


# -- get_recent_for_project ---------------------------------------------------

def test_reuses_cache_for_a_near_identical_reask():
    reg = _completed_registry_with(
        "Qual è lo stato del progetto?",
        "Sir, tutto procede secondo i piani.",
    )
    recent = reg.get_recent_for_project("jarvis", "Qual è lo stato attuale del progetto?")
    assert recent is not None
    assert recent["summary"] == "Sir, tutto procede secondo i piani."


def test_status_check_then_different_work_request(caplog):
    """The exact scenario from the incident: a status-check dispatch
    completes, then — well inside the 5-minute cache window — the user asks
    for unrelated new work on the same project. The cache must not answer
    the second question with the first's answer.
    """
    reg = _completed_registry_with(
        "Rivedi lo stato attuale del progetto. Ci sono diverse richieste in "
        "sospeso su Obsidian come sistema di memoria per JARVIS — dimmi cosa "
        "è stato deciso o implementato finora, e qual è lo stato attuale.",
        "Maestro, ho revisionato lo stato di Obsidian in JARVIS. Nel primo "
        "livello, già mergiato, c'è `obsidian_access.py` che legge il vault "
        "in sola lettura per cercare note—niente modifiche per design. Nel s",
    )
    new_prompt = (
        "Fai commit di memory_vault.py e integralo in server.py così che "
        "JARVIS scriva effettivamente la memoria sul vault Obsidian in "
        "JARVIS/Memoria/. Conferma quando fatto."
    )
    recent = reg.get_recent_for_project("jarvis", new_prompt)
    assert recent is None, "must not silently answer a new request with the old dispatch's summary"


def test_no_completed_dispatch_returns_none():
    reg = dr.DispatchRegistry()
    reg.register("jarvis", "/path", "Some prompt")  # left pending, never completed
    assert reg.get_recent_for_project("jarvis", "Some prompt") is None


def test_outside_max_age_returns_none_even_with_matching_prompt(monkeypatch):
    reg = _completed_registry_with("Qual è lo stato?", "Tutto ok.")
    # Force the stored completion far enough in the past to fall outside a
    # short window, independent of prompt similarity.
    conn = dr._get_db()
    conn.execute("UPDATE dispatches SET completed_at = ?", (time.time() - 3600,))
    conn.commit()
    conn.close()
    recent = reg.get_recent_for_project("jarvis", "Qual è lo stato?", max_age_seconds=60)
    assert recent is None
