"""JARVIS memory → Obsidian mirror.

SQLite (`memory.py`) stays the one source of truth `build_memory_context()`
reads from on every turn — that path must stay a fast FTS5 query, not a
markdown parse. This module is a one-way mirror off to the side: every
`remember()` also appends a line to a note in the user's own vault, so what
JARVIS knows is something a person can actually open, read, and recognize
as belonging to their own tools — not a row hidden in a `.db` file.

Deliberately one-way. Nothing here reads the vault back into memory: a
memory becomes a fact because the extraction pipeline in `memory.py` decided
it was one, with the supersede logic that entails. A stray edit made
directly in Obsidian must not silently become a new fact with none of that
— so this module is pure output. `obsidian_access.py` (search/read the
user's actual notes) stays read-only for the same reason it already was:
nothing here touches it, or anything outside its own dedicated folder.

Every public function fails closed: no `OBSIDIAN_VAULT` configured, or any
write error, and the mirror is silently skipped. It is a nice-to-have next
to the real memory in SQLite, never a dependency of it — the log message
you'd see for a failed mirror write is at most a warning.
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

_HEADER = (
    "# {title}\n\n"
    "Generato e aggiornato automaticamente da JARVIS — rispecchia la sua "
    "memoria interna. Modifiche fatte qui non vengono lette da JARVIS: la "
    "fonte resta il suo database; questa pagina è la finestra su di esso, "
    "non un ingresso.\n\n"
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
