"""confirm.py — a confirmation gate the model cannot forge.

The old shape of a "dangerous action" gate is a tool parameter:

    if action in DANGEROUS_ACTIONS:
        confirmed = params.get("confirmed", False)

`confirmed` there is written by the model, not the user — nothing stops an
LLM from sending confirmed=true on the very first call. It is a convention,
not a gate.

The design here instead has the *interface* issue the token:

    1. Something irreversible calls `request(key, detail, run)` and returns
       immediately — `run` (sync or async, zero-arg) is not called yet.
    2. The caller replies to the user right away; nothing blocks on this.
    3. The UI shows a CONFIRM / CANCEL banner built from the returned
       payload. Only a person clicking CONFIRM calls `resolve(key, True)`,
       which is what actually runs `run`.

A pending confirmation nobody answers is abandoned after TIMEOUT_SECONDS, so
a stale banner can never authorize a click made long after it was asked.

What belongs here: things that cannot be reversed once started (a server
restart is already underway the moment it runs). Anything reversible should
run at once and be pushed onto undo.py instead — asking before an undoable
action is a worse experience for no extra safety.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Union

log = logging.getLogger("jarvis.confirm")

TIMEOUT_SECONDS = 90.0

RunFn = Callable[[], Union[str, Awaitable[str]]]


@dataclass
class _Pending:
    key: str
    detail: str
    run: RunFn
    at: float


_pending: Optional[_Pending] = None
_lock = asyncio.Lock()


async def request(key: str, detail: str, run: RunFn) -> dict:
    """Register a pending irreversible action and return the banner payload.

    Replaces whatever was previously pending — only one confirmation is ever
    live, matching the single-banner UI.
    """
    global _pending
    async with _lock:
        _pending = _Pending(key=key, detail=detail, run=run, at=time.time())
    log.info(f"Confirmation requested: {key} — {detail}")
    return {"key": key, "detail": detail, "timeout": TIMEOUT_SECONDS}


async def resolve(key: str, confirmed: bool) -> Optional[str]:
    """Run (if confirmed) or discard the pending action matching `key`.

    Returns the result string from `run` on success, or None when there was
    nothing pending, the key didn't match the live banner, it had already
    expired, or the user cancelled.
    """
    global _pending
    async with _lock:
        pending = _pending
        if pending is None or pending.key != key:
            return None
        expired = time.time() - pending.at > TIMEOUT_SECONDS
        _pending = None

    if expired:
        log.info(f"Confirmation expired before resolution: {key}")
        return None
    if not confirmed:
        log.info(f"Confirmation cancelled: {key}")
        return None

    log.info(f"Confirmation approved, running: {key}")
    result = pending.run()
    if inspect.isawaitable(result):
        result = await result
    return result
