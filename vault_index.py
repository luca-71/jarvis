"""Semantic recall over the user's Obsidian vault, embedded locally by Ollama.

`memory.py` answers "what did JARVIS decide to remember", by keyword. This
answers a different question — "has the user already written about this?" —
over 113 notes nobody is going to tag or curate by hand. Keyword search misses
it: a note titled "Lupiae — gestionale cimiteriale" never contains the words
"quel lavoro per il comune", and FTS5 cannot bridge that. Embeddings can.

Local on purpose. nomic-embed-text is 274 MB and turns a sentence into 768
numbers in about 25ms on this machine — no API call, no token cost, no credit
balance to run out of. It is the one job on an 8 GB M1 where a local model
genuinely wins: not generating text, indexing it.

Read-only, always. This module never writes to the vault. The mirror in the
other direction is memory_vault.py, and JARVIS/Memoria is excluded from the
index here — indexing the mirror would feed JARVIS's own memories back as if
they were something the user had written.
"""

import array
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("jarvis.vault_index")

DB_PATH = Path(__file__).parent / "data" / "jarvis.db"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("JARVIS_EMBED_MODEL", "nomic-embed-text")

# Folders never indexed. JARVIS/ holds the memory mirror — see the docstring —
# and .obsidian is the app's own config.
_SKIP_DIRS = {".obsidian", ".trash", ".git", "JARVIS", "node_modules"}


def _excluded() -> set[str]:
    """Note names or folders the user has asked to keep out of the index.

    A vault holds things a voice assistant should not surface unprompted — this
    one has a note on the user's finances, which semantic search happily
    volunteered for questions about pasta before the threshold was calibrated.
    Case-insensitive substring match against the path, so "profilo" excludes
    profilo.md and "Privato/" excludes a whole folder.
    """
    raw = os.getenv("JARVIS_VAULT_EXCLUDE", "")
    return {p.strip().lower() for p in raw.split(",") if p.strip()}

# Chunks are built to sit under this many characters. Long enough to keep a
# thought intact, short enough that a hit points at a paragraph rather than at
# "somewhere in this 8 KB note".
_MAX_CHUNK = 900
_MIN_CHUNK = 60

# Similarity below this is not a match. Measured on this vault rather than
# guessed, because the first guess (0.55) let everything through: questions the
# notes say nothing about — "ricetta della carbonara", "storia dell'impero
# romano" — still scored up to 0.614, and the chunk they kept dragging in was
# `profilo > Situazione attuale`, which holds the user's finances. Real hits sit
# at 0.70+. 0.65 is the gap between the two.
#
# Re-measure this if the embedding model changes: bge-m3 separates just as well
# but on a lower scale entirely (hits ~0.52, noise ~0.46), so this number would
# silently reject everything.
#
# It does not separate perfectly and no single number would. "che tempo fa
# domani" scores 0.688, above "cosa so di lupiae" at 0.667 — a real question
# about the vault ranking below an unrelated one. Scoring by how far the top
# result stands out from the pack was measured as an alternative and separates
# worse (relevant +0.022..+0.111, irrelevant +0.014..+0.035 — overlapping), so
# it is not worth the machinery. The residual cost is an occasional stray
# paragraph in the prompt, which the model ignores.
DEFAULT_THRESHOLD = 0.65

# Never more than this in a prompt, however many clear the bar.
DEFAULT_TOP_K = 3

_EMBED_BATCH = 32

# nomic-embed-text documents `search_document:` / `search_query:` prefixes, and
# adding them measurably made this vault WORSE: the correct hit for "quel
# gestionale che ho fatto per il comune" fell from 0.70 to 0.66 and dropped to
# third, while the noise floor stayed at 0.59 — the signal-to-noise gap halved.
# The tag installed here evidently predates that convention. Left empty rather
# than deleted, so the next person to read the docs and reach for them finds
# the measurement instead of repeating it.
_DOC_PREFIX = ""
_QUERY_PREFIX = ""


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS vault_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            title TEXT NOT NULL,
            heading TEXT DEFAULT '',
            content TEXT NOT NULL,
            vector BLOB NOT NULL,
            indexed_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_vault_chunks_path ON vault_chunks(path);

        -- One row per file, so re-indexing can skip anything unchanged.
        CREATE TABLE IF NOT EXISTS vault_files (
            path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            digest TEXT NOT NULL,
            chunks INTEGER NOT NULL,
            indexed_at REAL NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def _pack(vector: list[float]) -> bytes:
    return array.array("f", vector).tobytes()


def _unpack(blob: bytes) -> array.array:
    vec = array.array("f")
    vec.frombytes(blob)
    return vec


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed(texts: list[str], timeout: float = 120) -> list[list[float]] | None:
    """Embed a batch through Ollama. None when Ollama is unreachable.

    Fails soft everywhere it is called: an assistant whose voice stops working
    because a background index could not reach a local daemon would be a bad
    trade for a feature that only ever adds context.
    """
    if not texts:
        return []
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response).get("embeddings")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:200]
        log.warning(f"Embedding failed ({e.code}): {detail}")
    except Exception as e:
        log.warning(f"Ollama unreachable for embeddings: {e}")
    return None


def _cosine(a: array.array, b: array.array) -> float:
    """Plain Python: 768 dimensions over a few hundred chunks is a few
    milliseconds, and numpy is not a dependency of this project."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def _chunk_note(text: str) -> list[tuple[str, str]]:
    """Split a note into (heading, content) pairs.

    Headings are carried into the chunk text as well as stored: "Prezzi" as a
    bare heading means nothing to an embedding, but "Lupiae Servizi > Prezzi"
    followed by the paragraph does.
    """
    text = _FRONTMATTER_RE.sub("", text)
    sections: list[tuple[str, str]] = []

    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        sections.append(("", text))
    else:
        if matches[0].start() > 0:
            sections.append(("", text[:matches[0].start()]))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((match.group(2).strip(), text[match.end():end]))

    chunks: list[tuple[str, str]] = []
    for heading, body in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buffer = ""
        for paragraph in paragraphs:
            if len(buffer) + len(paragraph) + 2 <= _MAX_CHUNK:
                buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
                continue
            if buffer:
                chunks.append((heading, buffer))
            # A single paragraph over the limit is cut on whitespace rather
            # than mid-word; oversized paragraphs are rare and rarely dense.
            while len(paragraph) > _MAX_CHUNK:
                cut = paragraph.rfind(" ", 0, _MAX_CHUNK)
                cut = cut if cut > _MAX_CHUNK // 2 else _MAX_CHUNK
                chunks.append((heading, paragraph[:cut]))
                paragraph = paragraph[cut:].lstrip()
            buffer = paragraph
        if buffer:
            chunks.append((heading, buffer))

    return [(h, c) for h, c in chunks if len(c.strip()) >= _MIN_CHUNK]


def _vault_root() -> Path | None:
    raw = os.getenv("OBSIDIAN_VAULT", "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser()
    return root if root.is_dir() else None


def _notes(root: Path):
    excluded = _excluded()
    for path in root.rglob("*.md"):
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if excluded and any(term in str(rel).lower() for term in excluded):
            continue
        yield path


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def reindex(force: bool = False) -> dict:
    """Index every changed note. Returns a summary dict.

    Blocking and CPU-light — the waiting is on Ollama. Callers run it in a
    thread.
    """
    root = _vault_root()
    if root is None:
        return {"status": "no_vault", "indexed": 0, "skipped": 0, "chunks": 0}

    init_db()
    conn = _db()
    known = {
        row["path"]: (row["mtime"], row["digest"])
        for row in conn.execute("SELECT path, mtime, digest FROM vault_files")
    }

    seen: set[str] = set()
    indexed = skipped = total_chunks = 0
    started = time.time()

    for path in _notes(root):
        rel = str(path.relative_to(root))
        seen.add(rel)
        try:
            stat = path.stat()
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.debug(f"Skipping {rel}: {e}")
            continue

        digest = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()
        previous = known.get(rel)
        # mtime alone re-indexes on every touch; the digest is what makes a
        # save with no edits free.
        if not force and previous and previous[1] == digest:
            skipped += 1
            continue

        chunks = _chunk_note(raw)
        if not chunks:
            conn.execute("DELETE FROM vault_chunks WHERE path = ?", (rel,))
            conn.execute(
                "INSERT OR REPLACE INTO vault_files (path, mtime, digest, chunks, indexed_at) "
                "VALUES (?, ?, ?, 0, ?)", (rel, stat.st_mtime, digest, time.time()))
            continue

        title = path.stem
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), _EMBED_BATCH):
            batch = chunks[start:start + _EMBED_BATCH]
            payload = [
                _DOC_PREFIX + (f"{title} > {heading}\n{content}" if heading
                               else f"{title}\n{content}")
                for heading, content in batch
            ]
            got = _embed(payload)
            if got is None:
                conn.commit()
                conn.close()
                log.warning("Reindex aborted — embeddings unavailable")
                return {"status": "embeddings_unavailable", "indexed": indexed,
                        "skipped": skipped, "chunks": total_chunks}
            vectors.extend(got)

        # Replace wholesale: a note's chunks shift as it is edited, so
        # reconciling them individually would be work for no benefit.
        conn.execute("DELETE FROM vault_chunks WHERE path = ?", (rel,))
        now = time.time()
        conn.executemany(
            "INSERT INTO vault_chunks (path, title, heading, content, vector, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(rel, title, heading, content, _pack(vector), now)
             for (heading, content), vector in zip(chunks, vectors)],
        )
        conn.execute(
            "INSERT OR REPLACE INTO vault_files (path, mtime, digest, chunks, indexed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (rel, stat.st_mtime, digest, len(chunks), now))
        indexed += 1
        total_chunks += len(chunks)

    # Notes deleted or renamed since the last pass.
    removed = [p for p in known if p not in seen]
    for rel in removed:
        conn.execute("DELETE FROM vault_chunks WHERE path = ?", (rel,))
        conn.execute("DELETE FROM vault_files WHERE path = ?", (rel,))

    conn.commit()
    conn.close()

    elapsed = time.time() - started
    log.info(f"Vault index: {indexed} notes ({total_chunks} chunks), "
             f"{skipped} unchanged, {len(removed)} removed, {elapsed:.1f}s")
    return {"status": "ok", "indexed": indexed, "skipped": skipped,
            "chunks": total_chunks, "removed": len(removed), "seconds": elapsed}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = DEFAULT_TOP_K,
           threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Chunks most similar to `query`, best first. Empty on any failure."""
    if not query or not query.strip():
        return []

    vectors = _embed([_QUERY_PREFIX + query.strip()], timeout=20)
    if not vectors:
        return []
    query_vec = array.array("f", vectors[0])

    try:
        conn = _db()
        rows = conn.execute(
            "SELECT path, title, heading, content, vector FROM vault_chunks"
        ).fetchall()
        conn.close()
    except sqlite3.Error as e:
        log.warning(f"Vault index unreadable: {e}")
        return []

    scored = []
    for row in rows:
        score = _cosine(query_vec, _unpack(row["vector"]))
        if score >= threshold:
            scored.append({
                "path": row["path"], "title": row["title"],
                "heading": row["heading"], "content": row["content"],
                "score": score,
            })

    scored.sort(key=lambda r: r["score"], reverse=True)

    # One chunk per note: three hits from the same file crowd out the second
    # relevant note, and the best chunk already carries the point.
    best_per_note: dict[str, dict] = {}
    for hit in scored:
        best_per_note.setdefault(hit["path"], hit)

    return list(best_per_note.values())[:top_k]


def format_for_prompt(hits: list[dict]) -> str:
    """Render hits as a prompt block, or '' when there is nothing to add."""
    if not hits:
        return ""
    lines = []
    for hit in hits:
        where = f"{hit['title']} > {hit['heading']}" if hit["heading"] else hit["title"]
        body = " ".join(hit["content"].split())
        if len(body) > 500:
            body = body[:500].rsplit(" ", 1)[0] + "…"
        lines.append(f"- [{where}] {body}")
    return (
        "FROM THE USER'S OWN NOTES (their Obsidian vault — they wrote this, "
        "you did not; cite the note when you use it):\n" + "\n".join(lines)
    )


def stats() -> dict:
    try:
        init_db()
        conn = _db()
        files = conn.execute("SELECT COUNT(*) AS n FROM vault_files").fetchone()["n"]
        chunks = conn.execute("SELECT COUNT(*) AS n FROM vault_chunks").fetchone()["n"]
        last = conn.execute("SELECT MAX(indexed_at) AS t FROM vault_files").fetchone()["t"]
        conn.close()
        return {"files": files, "chunks": chunks, "last_indexed": last}
    except sqlite3.Error as e:
        return {"files": 0, "chunks": 0, "last_indexed": None, "error": str(e)}
