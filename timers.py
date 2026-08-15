"""JARVIS timers — countdowns that survive a restart.

A timer is the one thing JARVIS is asked to remember that has a deadline
attached, which makes in-memory state exactly wrong for it: the server is
restarted several times an hour during development, and an asyncio.sleep task
dies with it silently. Nobody notices until the pasta is ruined. So timers live
in the same SQLite file as memory and dispatches, and the scheduler rebuilds
itself from the table on every start.

Persistence buys catch-up too. A timer that came due while the server was down
still fires when it comes back, with `late` set so the announcement can say so
rather than pretending it was punctual. Past MISSED_AFTER_SECONDS that stops
being helpful and starts being confusing — a timer from yesterday going off at
breakfast — so those are marked `missed` and never spoken; they stay in the
table for the LLM to mention if asked.

This module owns scheduling and storage only. It never imports the server, and
it does not know what speaking is: the caller injects an `on_fire` coroutine.
That keeps the fire path testable without a WebSocket, a voice, or an API key.
"""

import asyncio
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("jarvis.timers")

DB_PATH = Path(__file__).parent / "data" / "jarvis.db"

# A timer overdue by more than this when the server comes back is reported as
# missed rather than announced. An hour is long enough to cover a restart or a
# crash-and-fix, short enough that nothing rings at breakfast about yesterday.
MISSED_AFTER_SECONDS = 60 * 60

# Ceiling on how long the scheduler will sleep in one go. The loop wakes early
# whenever a timer is added, so this only bounds the damage from a wall-clock
# jump — a laptop resuming from sleep, mainly, where the deadline may have
# passed while the loop was frozen mid-sleep.
MAX_SLEEP_SECONDS = 60.0

# How often to re-check for a listener while holding a due timer back. Short,
# because this only runs in the seconds between the server accepting
# connections and a client making one.
LISTENER_POLL_SECONDS = 2.0

# Guard against a fat-fingered "[ACTION:TIMER] 36000000". Anything longer than a
# day belongs in the calendar, which JARVIS already has.
MAX_DURATION_SECONDS = 24 * 60 * 60


@dataclass
class Timer:
    id: int
    label: str
    fire_at: float
    created_at: float
    status: str

    @property
    def remaining(self) -> float:
        """Seconds left. Negative once overdue."""
        return self.fire_at - time.time()


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_timer_db():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS timers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            fire_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            fired_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_timers_pending ON timers(status, fire_at);
    """)
    conn.commit()
    conn.close()


def _row_to_timer(row: sqlite3.Row) -> Timer:
    return Timer(
        id=row["id"],
        label=row["label"] or "",
        fire_at=row["fire_at"],
        created_at=row["created_at"],
        status=row["status"],
    )


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

# The LLM is asked for plain seconds, which is the one format that survives
# translation — "dieci minuti", "ten minutes" and "十分" all become 600 before
# they reach this module. The suffix forms below are a safety net for when it
# writes "10m" anyway, not a natural-language parser: teaching this function
# every language JARVIS speaks is the LLM's job, and it already did it.
_UNIT_SECONDS = {"h": 3600, "m": 60, "s": 1}
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([hms])", re.IGNORECASE)


def parse_seconds(raw: str) -> float | None:
    """Seconds from `raw`, or None if it is not a duration.

    Accepts a bare number ("600", "600.0") or suffixed parts ("10m", "1h30m",
    "90s"). Returns None rather than a default, so a malformed tag declines to
    set a timer instead of setting a surprising one.
    """
    text = (raw or "").strip().lower()
    if not text:
        return None

    try:
        seconds = float(text)
    except ValueError:
        parts = _DURATION_RE.findall(text)
        if not parts:
            return None
        # Reject trailing junk: "10m of pasta" is a label that lost its
        # delimiter, not a duration, and should not silently become 600.
        if _DURATION_RE.sub("", text).strip():
            return None
        seconds = sum(float(value) * _UNIT_SECONDS[unit.lower()] for value, unit in parts)

    if seconds <= 0 or seconds > MAX_DURATION_SECONDS:
        return None
    return seconds


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def create_timer(seconds: float, label: str = "") -> Timer:
    """Store a timer due `seconds` from now."""
    now = time.time()
    fire_at = now + seconds
    conn = _get_db()
    cur = conn.execute(
        "INSERT INTO timers (label, created_at, fire_at, status) VALUES (?, ?, ?, 'pending')",
        (label.strip()[:100], now, fire_at),
    )
    timer_id = cur.lastrowid
    conn.commit()
    conn.close()
    log.info(f"Timer #{timer_id} set for {int(seconds)}s: {label or '(no label)'}")
    return Timer(id=timer_id, label=label.strip(), fire_at=fire_at, created_at=now, status="pending")


def active_timers() -> list[Timer]:
    """Pending timers, soonest first."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM timers WHERE status = 'pending' ORDER BY fire_at ASC"
    ).fetchall()
    conn.close()
    return [_row_to_timer(r) for r in rows]


def cancel_timer(timer_id: int) -> bool:
    """Cancel one pending timer. False if it was already fired or gone."""
    conn = _get_db()
    cur = conn.execute(
        "UPDATE timers SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
        (timer_id,),
    )
    changed = cur.rowcount > 0
    conn.commit()
    conn.close()
    return changed


def cancel_matching(query: str) -> list[Timer]:
    """Cancel pending timers whose label matches `query`.

    An empty query cancels every pending timer — "cancel my timers" is a whole
    sentence in every language and not worth parsing more finely than this.
    """
    query = (query or "").strip()
    candidates = active_timers()
    if query:
        needle = query.lower()
        candidates = [t for t in candidates if needle in t.label.lower()]
    return [t for t in candidates if cancel_timer(t.id)]


def _claim_due(now: float) -> list[tuple[Timer, bool]]:
    """Atomically take ownership of every timer now due.

    Each row moves out of 'pending' in the same statement that selects it, so a
    timer fires exactly once even if two schedulers are somehow running. The
    bool is whether the timer is late enough to say so; rows past
    MISSED_AFTER_SECONDS are marked 'missed' and are not returned at all —
    catching up on those out loud is noise, not service.
    """
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM timers WHERE status = 'pending' AND fire_at <= ? ORDER BY fire_at ASC",
        (now,),
    ).fetchall()

    claimed = []
    for row in rows:
        overdue = now - row["fire_at"]
        status = "missed" if overdue > MISSED_AFTER_SECONDS else "fired"
        cur = conn.execute(
            "UPDATE timers SET status = ?, fired_at = ? WHERE id = ? AND status = 'pending'",
            (status, now, row["id"]),
        )
        if cur.rowcount and status == "fired":
            # Anything more than a few seconds behind was slept through, not
            # merely rounded — that is worth admitting when it announces.
            claimed.append((_row_to_timer(row), overdue > 5))
        elif cur.rowcount:
            log.info(f"Timer #{row['id']} missed by {int(overdue)}s — not announcing")

    conn.commit()
    conn.close()
    return claimed


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TimerService:
    """Fires due timers by calling `on_fire(timer, late)`.

    One task, no task-per-timer: a hundred pending timers are a hundred rows
    and one sleep, and cancelling is an UPDATE rather than chasing a handle.

    `has_listener` guards the claim. Claiming is irreversible — the row leaves
    'pending' and can never fire again — so doing it while nobody is connected
    destroys the announcement instead of delivering it. That is not a rare
    race: catch-up timers come due during startup by definition, and the
    scheduler starts a few hundred milliseconds before the first client
    connects, which is exactly long enough to lose one.
    """

    def __init__(self, on_fire, has_listener=None):
        self._on_fire = on_fire
        self._has_listener = has_listener
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self):
        init_timer_db()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
            log.info("Timer scheduler started")

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()

    def notify(self):
        """Wake the loop early — a timer was added or cancelled."""
        self._wake.set()

    def _listening(self) -> bool:
        if self._has_listener is None:
            return True
        try:
            return bool(self._has_listener())
        except Exception:
            return True

    async def _run(self):
        while True:
            try:
                listening = self._listening()
                if listening:
                    for timer, late in _claim_due(time.time()):
                        # One timer's announcement must not swallow the next.
                        try:
                            await self._on_fire(timer, late)
                        except Exception as e:
                            log.error(f"Timer #{timer.id} fired but could not be announced: {e}")

                pending = active_timers()
                if not pending:
                    delay = MAX_SLEEP_SECONDS
                elif pending[0].remaining <= 0 and not listening:
                    # Due, but there is nobody to tell. Hold it — still pending,
                    # so it announces the moment someone connects, by which
                    # point it is late and will say so.
                    delay = LISTENER_POLL_SECONDS
                else:
                    delay = max(0.0, min(pending[0].remaining, MAX_SLEEP_SECONDS))

                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A scheduler that dies takes every future timer with it, so no
                # unexpected error is allowed to leave the loop.
                log.error(f"Timer scheduler error: {e}")
                await asyncio.sleep(5)


def format_for_prompt() -> str:
    """Pending timers as context for the LLM.

    Remaining time is given in seconds rather than prose because the reply is
    spoken in whatever language is configured, and only the model can phrase
    "four and a half minutes" correctly in all of them.
    """
    pending = active_timers()
    if not pending:
        return "No timers running."
    lines = []
    for t in pending:
        label = t.label or "unlabelled"
        lines.append(f"  - #{t.id} {label}: {int(max(0, t.remaining))}s remaining")
    return "TIMERS:\n" + "\n".join(lines)
