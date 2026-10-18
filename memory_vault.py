"""JARVIS memory <-> Obsidian vault, kept in both directions.

SQLite (`memory.py`) is still what `build_memory_context()` reads on every
turn — that path stays a fast FTS5 query, never a markdown parse at
conversation time. But the vault is the authoritative record: every
`remember()` appends a line to a note in the user's own vault, and
`sync_from_vault()` (called hourly from the same loop that reindexes the
user's notes — see `_index_vault()` in server.py) reads those notes back and
reconciles SQLite to match. A line the user typed by hand becomes a real
memory; an edit to a line's text overwrites what's on file; deleting a line
forgets it. SQLite is the fast index; the vault is the source it's indexing.

This is a deliberate reversal of this module's original one-way design,
which existed specifically so a stray Obsidian edit could never sneak in as
an unreviewed fact. That protection is gone now by choice: the trade is
supersede/correction bookkeeping (a hand-added line has no way to say what
it replaces — it can only ever add or flatly overwrite) for a vault that is
actually the thing JARVIS knows, editable from any device Obsidian syncs to,
not a read-only window onto a database file. `obsidian_access.py`
(search/read the user's actual notes) stays read-only regardless — nothing
here touches it, or anything outside this module's own dedicated folder.

Every public function fails closed: no `OBSIDIAN_VAULT` configured, or any
read/write error, and the sync step for that file is silently skipped. A
vault sync failing costs the next hour's reconciliation, never the memory
already committed in SQLite.
"""

import logging
import os
import re
import time
from pathlib import Path

log = logging.getLogger("jarvis.memory_vault")

# Filenames are Italian because the vault they land in is — matching the
# user's own notes rather than the app's internal (English) type names.
_TYPE_FILENAMES = {
    "fact": "Fatti.md",
    "preference": "Preferenze.md",
    "project": "Progetti.md",
    "person": "Persone.md",
    "decision": "Decisioni.md",
}
_FALLBACK_FILENAME = "Altro.md"

# In-process only — read by the HUD tile (`/api/hud/stats`) to show when
# sync last ran and what it did, without making that endpoint touch the
# filesystem itself on every poll.
_last_sync: dict = {"status": "never_run", "at": None, "added": 0, "updated": 0, "removed": 0}


def get_last_sync() -> dict:
    return dict(_last_sync)

_HEADER = (
    "# {title}\n\n"
    "Generato da JARVIS e riletto da lui ogni ora — è la sua memoria reale, "
    "non solo una finestra su di essa. Aggiungi una riga per fargli imparare "
    "un fatto nuovo, modifica il testo di una riga per correggerlo, elimina "
    "una riga per fargliela dimenticare. Non toccare `<!-- id:N -->`: è "
    "quello che lega la riga alla voce nel suo database.\n\n"
)


def _vault_memory_dir() -> Path | None:
    """The dedicated, JARVIS-owned folder inside the vault — never the vault
    root, and never anywhere the user's own notes already live. Same
    isolation `notes_access.py` and the rest of this app already use: a
    voice misfire can make a mess of a folder JARVIS owns, not of one it
    doesn't."""
    raw = os.getenv("OBSIDIAN_VAULT", "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser()
    if not root.is_dir():
        return None
    memory_dir = root / "JARVIS" / "Memoria"
    try:
        memory_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning(f"Could not create vault memory folder: {e}")
        return None
    return memory_dir


def _file_for_type(mem_type: str) -> str:
    return _TYPE_FILENAMES.get(mem_type, _FALLBACK_FILENAME)


def _read_or_init(path: Path, mem_type: str) -> str:
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            pass
    title = _file_for_type(mem_type).removesuffix(".md")
    return _HEADER.format(title=title)


def mirror_remember(mem_id: int, mem_type: str, content: str) -> None:
    """Append one memory as a line, tagged with its SQLite id.

    The id marker is what makes `mirror_forget` possible later — an id-less
    line could only be matched by its text, which breaks the moment a memory
    is superseded by near-identical wording (exactly the case that matters:
    a corrected fact usually reads a lot like the fact it's correcting).
    """
    memory_dir = _vault_memory_dir()
    if memory_dir is None:
        return

    path = memory_dir / _file_for_type(mem_type)
    try:
        body = _read_or_init(path, mem_type)
        stamp = time.strftime("%Y-%m-%d %H:%M")
        # Content is free-form model output — a stray newline would silently
        # split one memory into a fragment and an orphaned second bullet.
        clean_content = " ".join(content.split())
        line = f"- [{stamp}] {clean_content} <!-- id:{mem_id} -->\n"
        path.write_text(body + line, encoding="utf-8")
    except OSError as e:
        log.warning(f"Vault mirror write failed for memory #{mem_id}: {e}")


_ID_LINE_RE_TEMPLATE = r"^- .*<!-- id:{id} -->\n?"


def mirror_forget(mem_id: int, mem_type: str | None = None) -> None:
    """Remove one memory's line, wherever it landed.

    `mem_type` narrows the search to one file when the caller has it (the
    common case); without it every type file is checked, since a memory
    forgotten generically (`forget()` alone knows only the id) can't know
    which file it was mirrored into without re-deriving it.
    """
    memory_dir = _vault_memory_dir()
    if memory_dir is None:
        return

    filenames = [_file_for_type(mem_type)] if mem_type else list(_TYPE_FILENAMES.values()) + [_FALLBACK_FILENAME]
    pattern = re.compile(_ID_LINE_RE_TEMPLATE.format(id=mem_id), re.MULTILINE)

    for filename in filenames:
        path = memory_dir / filename
        if not path.exists():
            continue
        try:
            body = path.read_text(encoding="utf-8")
            new_body, count = pattern.subn("", body)
            if count:
                path.write_text(new_body, encoding="utf-8")
                return  # An id is written to exactly one file; done once found.
        except OSError as e:
            log.warning(f"Vault mirror removal failed for memory #{mem_id} in {filename}: {e}")
            return


_BULLET_ID_RE = re.compile(r"<!--\s*id:(\d+)\s*-->")
_BULLET_STAMP_RE = re.compile(r"^\[[^\]]*\]\s*")


def _parse_bullets(body: str) -> list[tuple[int, str | None, str]]:
    """One entry per `- ...` line: (line index, id or None, content).

    An id comment is what `mirror_remember` stamps on — its absence is the
    signal that a line was typed by hand in Obsidian, not written by JARVIS.
    """
    entries = []
    for i, raw_line in enumerate(body.splitlines()):
        stripped = raw_line.strip()
        if not stripped.startswith("- "):
            continue
        text = stripped[2:].strip()
        id_match = _BULLET_ID_RE.search(text)
        mem_id = int(id_match.group(1)) if id_match else None
        if id_match:
            text = text[:id_match.start()].strip()
        text = _BULLET_STAMP_RE.sub("", text).strip()
        if text:
            entries.append((i, mem_id, text))
    return entries


def sync_from_vault() -> dict:
    """Reconcile SQLite to what the vault's memory notes actually say.

    Called hourly (see `_index_vault()` in server.py) — not on a watcher, for
    the same reason note reindexing isn't: a vault that syncs across devices
    settles faster than anything meaningful actually changes.

    Per type file: a bulleted line with no id is new and gets one stamped
    back into the file; a line whose id is on file but whose text no longer
    matches SQLite is an edit and overwrites it there.

    Deletion is gated behind `vault_synced` (see `memory.mark_vault_synced`)
    and deliberately conservative: a memory only becomes eligible to be
    deleted for "the file no longer lists it" once this function (or a
    successful mirror write) has actually seen it there at least once.
    Without that gate, every memory that predates vault mirroring — or whose
    mirror write once failed — is absent from every file by definition, and
    reads as "the user deleted this" on the very first run. That is not
    hypothetical: it is what the first version of this function did, and it
    erased nine real memories the first time it ran, the user's own name and
    city among them, before the flag existed to stop it.
    """
    memory_dir = _vault_memory_dir()
    if memory_dir is None:
        _last_sync.update(status="unconfigured", at=time.time())
        return {"status": "unconfigured"}

    import memory as _memory  # deferred: memory.py imports this module first

    added = updated = removed = 0
    stamp = time.strftime("%Y-%m-%d %H:%M")

    for mem_type, filename in list(_TYPE_FILENAMES.items()) + [(None, _FALLBACK_FILENAME)]:
        path = memory_dir / filename
        if not path.exists():
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning(f"Vault sync read failed for {filename}: {e}")
            continue

        # "other" (not one of the five real types) rather than defaulting a
        # hand-added Altro.md line to "fact": that would make it collect in
        # Fatti.md's own deletion pass on the next run — the id would exist in
        # SQLite as type=fact but never appear inside Fatti.md itself, reading
        # as a deletion that never happened.
        type_key = mem_type or "other"
        db_by_id = {row["id"]: row for row in _memory.get_all_memories(type_key)}

        entries = _parse_bullets(body)
        lines = body.splitlines()
        file_dirty = False
        file_ids: set[int] = set()

        for line_index, mem_id, content in entries:
            if mem_id is None:
                new_id = _memory.remember(
                    content=content, mem_type=type_key, source="obsidian",
                    mirror=False, vault_synced=True,
                )
                lines[line_index] = f"- [{stamp}] {content} <!-- id:{new_id} -->"
                file_dirty = True
                file_ids.add(new_id)
                added += 1
            else:
                file_ids.add(mem_id)
                row = db_by_id.get(mem_id)
                if row is not None:
                    if not row["vault_synced"]:
                        # First confirmed sighting — earns the flag now, but
                        # not eligibility for THIS run's deletion pass below
                        # (db_by_id already has the pre-update value): a
                        # legacy id needs to have been seen present at least
                        # once, on some earlier run, before its absence on a
                        # later one means anything.
                        _memory.mark_vault_synced(mem_id)
                    if row["content"].strip() != content.strip():
                        _memory.update_memory_content(mem_id, content)
                        updated += 1

        if file_dirty:
            try:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            except OSError as e:
                log.warning(f"Vault sync write failed for {filename}: {e}")

        # Deletions: only rows already confirmed vault-backed (see the
        # function docstring) and that this file no longer lists.
        for mem_id, row in db_by_id.items():
            if row["vault_synced"] and mem_id not in file_ids:
                _memory.forget(mem_id)
                removed += 1

    _last_sync.update(status="ok", at=time.time(), added=added, updated=updated, removed=removed)

    if added or updated or removed:
        log.info(f"Vault memory sync: +{added} ~{updated} -{removed}")
    return {"status": "ok", "added": added, "updated": updated, "removed": removed}
