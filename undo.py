"""undo.py — takes back what JARVIS just did.

Actions that create or change something (a memory, a task, a note) register
their reversal here as a plain closure. Saying "undo" — in any language, the
LLM maps it onto [ACTION:UNDO] — pops the most recent one and runs it.

Deliberately NOT in scope: anything already covered by the confirm.py gate.
That module exists for actions that can't be undone (a server restart is
already in flight the moment it runs); this one is for actions that can, and
should therefore never make the user stop and answer a question first.

The stack is process memory only. A restart clears it — there is nothing to
undo across a restart anyway, since nothing about *this* running process's
actions survives one.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Union

log = logging.getLogger("jarvis.undo")

# A runaway guard, not a feature — normal use never approaches it. Unbounded
# would leak memory over a session that runs for days; a fixed cap costs
# nothing to enforce with deque(maxlen=...).
_MAX_ENTRIES = 50

UndoFn = Callable[[], Union[None, Awaitable[None]]]


@dataclass
class _Entry:
    label: str
    undo: UndoFn
    at: float


_stack: deque[_Entry] = deque(maxlen=_MAX_ENTRIES)


def register(label: str, undo: UndoFn) -> None:
    """Record how to reverse an action that just happened.

    `label` is what gets read back to the user afterwards ("Task added:
    Call the client"), so pass the same description already used for the
    activity log — not a code name.
    """
    _stack.append(_Entry(label=label, undo=undo, at=time.time()))


async def undo_last() -> Optional[str]:
    """Reverse the most recent registered action. Returns its label, or
    None if there was nothing to undo."""
    if not _stack:
        return None
    entry = _stack.pop()
    result = entry.undo()
    if inspect.isawaitable(result):
        await result
    log.info(f"Undid: {entry.label}")
    return entry.label
