"""Unit tests for memory_vault — the one-way SQLite → Obsidian memory mirror."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import memory_vault


@pytest.fixture(autouse=True)
def isolated_vault(tmp_path, monkeypatch):
    """Point every test at its own throwaway vault, never the real one."""
    monkeypatch.setenv("OBSIDIAN_VAULT", str(tmp_path))
    yield tmp_path


# -- vault resolution --------------------------------------------------------

def test_no_env_var_disables_mirror(monkeypatch):
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    memory_vault.mirror_remember(1, "fact", "content")
    # Nothing to assert on disk — the point is that this doesn't raise.


def test_nonexistent_vault_path_disables_mirror(monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT", "/nonexistent/path/xyz")
    memory_vault.mirror_remember(1, "fact", "content")


def test_creates_dedicated_subfolder_not_vault_root(isolated_vault):
    memory_vault.mirror_remember(1, "fact", "Something true")
    assert not (isolated_vault / "Fatti.md").exists(), "must not write into the vault root"
    assert (isolated_vault / "JARVIS" / "Memoria" / "Fatti.md").exists()


# -- writing -------------------------------------------------------------

def test_remember_writes_content_and_id_marker(isolated_vault):
    memory_vault.mirror_remember(42, "fact", "The user is called Luca.")
    body = (isolated_vault / "JARVIS" / "Memoria" / "Fatti.md").read_text()
    assert "The user is called Luca." in body
    assert "<!-- id:42 -->" in body


def test_each_type_goes_to_its_own_file(isolated_vault):
    memory_vault.mirror_remember(1, "fact", "A fact")
    memory_vault.mirror_remember(2, "preference", "A preference")
    memory_vault.mirror_remember(3, "project", "A project")
    memory_vault.mirror_remember(4, "person", "A person")
    memory_vault.mirror_remember(5, "decision", "A decision")
    memory_dir = isolated_vault / "JARVIS" / "Memoria"
    for name in ("Fatti.md", "Preferenze.md", "Progetti.md", "Persone.md", "Decisioni.md"):
        assert (memory_dir / name).exists(), f"missing {name}"


def test_unknown_type_falls_back_to_altro(isolated_vault):
    memory_vault.mirror_remember(1, "something_new", "Unclassified content")
    assert (isolated_vault / "JARVIS" / "Memoria" / "Altro.md").exists()


def test_multiple_memories_append_not_overwrite(isolated_vault):
    memory_vault.mirror_remember(1, "fact", "First fact")
    memory_vault.mirror_remember(2, "fact", "Second fact")
    body = (isolated_vault / "JARVIS" / "Memoria" / "Fatti.md").read_text()
    assert "First fact" in body
    assert "Second fact" in body


def test_multiline_content_collapsed_to_one_bullet(isolated_vault):
    """A stray newline in model output must not split one memory into a
    fragment plus an orphaned second bullet."""
    memory_vault.mirror_remember(1, "fact", "Line one\nLine two\n\nLine three")
    body = (isolated_vault / "JARVIS" / "Memoria" / "Fatti.md").read_text()
    content_lines = [l for l in body.splitlines() if l.startswith("- ")]
    assert len(content_lines) == 1
    assert "Line one Line two Line three" in content_lines[0]


def test_file_gets_a_header_on_first_write(isolated_vault):
    memory_vault.mirror_remember(1, "fact", "Something")
    body = (isolated_vault / "JARVIS" / "Memoria" / "Fatti.md").read_text()
    assert body.startswith("# Fatti")
    assert "non vengono lette da JARVIS" in body  # the read-only-mirror disclaimer


def test_header_not_duplicated_on_second_write(isolated_vault):
    memory_vault.mirror_remember(1, "fact", "First")
    memory_vault.mirror_remember(2, "fact", "Second")
    body = (isolated_vault / "JARVIS" / "Memoria" / "Fatti.md").read_text()
    assert body.count("# Fatti") == 1


# -- removal (forget / supersede) ----------------------------------------

def test_forget_removes_exactly_that_line(isolated_vault):
    memory_vault.mirror_remember(1, "fact", "Keep this")
    memory_vault.mirror_remember(2, "fact", "Remove this")
    memory_vault.mirror_forget(2, "fact")
    body = (isolated_vault / "JARVIS" / "Memoria" / "Fatti.md").read_text()
    assert "Keep this" in body
    assert "Remove this" not in body


def test_forget_without_type_searches_every_file(isolated_vault):
    """forget() in memory.py only always has an id — mirror_forget must be
    able to find the right file on its own when the caller doesn't know it."""
    memory_vault.mirror_remember(7, "preference", "Prefers dark mode")
    memory_vault.mirror_forget(7)  # no type given
    body = (isolated_vault / "JARVIS" / "Memoria" / "Preferenze.md").read_text()
    assert "Prefers dark mode" not in body


def test_forget_nonexistent_id_is_a_silent_no_op(isolated_vault):
    memory_vault.mirror_remember(1, "fact", "Something")
    memory_vault.mirror_forget(999)  # never existed
    body = (isolated_vault / "JARVIS" / "Memoria" / "Fatti.md").read_text()
    assert "Something" in body  # untouched


def test_supersede_by_near_identical_wording_removes_only_the_old_id(isolated_vault):
    """The exact failure mode text-matching would hit: a correction that
    reads almost like the fact it replaces."""
    memory_vault.mirror_remember(1, "fact", "The user's name is Martina.")
    memory_vault.mirror_forget(1, "fact")
    memory_vault.mirror_remember(2, "fact", "The user's name is Luca.")
    body = (isolated_vault / "JARVIS" / "Memoria" / "Fatti.md").read_text()
    assert "Martina" not in body
    assert "The user's name is Luca." in body
    assert "<!-- id:2 -->" in body


def test_removing_one_memory_leaves_others_and_header_intact(isolated_vault):
    memory_vault.mirror_remember(1, "fact", "Fact one")
    memory_vault.mirror_remember(2, "fact", "Fact two")
    memory_vault.mirror_remember(3, "fact", "Fact three")
    memory_vault.mirror_forget(2, "fact")
    body = (isolated_vault / "JARVIS" / "Memoria" / "Fatti.md").read_text()
    assert body.startswith("# Fatti")
    assert "Fact one" in body and "Fact three" in body
    assert "Fact two" not in body


# -- failure isolation -----------------------------------------------------

def test_write_failure_does_not_raise(isolated_vault, monkeypatch):
    """A vault write is a mirror, never a dependency — see memory.py's
    remember(), which wraps these calls but must never be broken by them."""
    memory_dir = isolated_vault / "JARVIS" / "Memoria"
    memory_dir.mkdir(parents=True)
    # Make the target file unwritable to force a real OSError on write.
    target = memory_dir / "Fatti.md"
    target.write_text("# Fatti\n\n")
    target.chmod(0o444)
    try:
        memory_vault.mirror_remember(1, "fact", "Should not raise")
    finally:
        target.chmod(0o644)  # restore so pytest can clean up tmp_path
