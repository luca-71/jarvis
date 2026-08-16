"""Subprocess helpers that never leak a child.

`asyncio.wait_for` cancels the *wait*, not the process. Every call site that
wrapped `proc.communicate()` in a timeout and then returned was leaving the
child alive forever — an `osascript` blocked on System Events (which is what
happens whenever the screen is locked or an app stops answering Apple events)
would sit there until the machine ran out of process slots. It did: after
roughly a day of idling, JARVIS hit `kern.maxprocperuid` and every fork after
that failed with EAGAIN, so TTS, work mode and calendar all broke at once.

`run_capture` is the only way this codebase should spawn a command it intends
to read the output of. It guarantees the child is dead and reaped before it
returns or raises, on every path — timeout, cancellation, or error.
"""

import asyncio
import logging
import os
import signal

log = logging.getLogger("jarvis.proc")

# AppleScript talking to a GUI app is the slowest thing we routinely run, and a
# hung one is indistinguishable from a slow one. Nothing may wait forever.
DEFAULT_TIMEOUT = 15.0

# Grace between SIGTERM and SIGKILL. AppleScript ignores SIGTERM while it is
# blocked inside an Apple event, so the kill is the one that usually lands.
_TERM_GRACE = 0.5


def env_without(*keys: str) -> dict[str, str]:
    """A copy of the current environment with `keys` removed.

    Computed fresh on every call, not cached — settings like
    ANTHROPIC_API_KEY can change at runtime (see /api/settings/keys), and a
    snapshot taken once at import time would keep handing out a stale value.
    """
    return {k: v for k, v in os.environ.items() if k not in keys}


def _signal_group(pid: int, sig: int) -> bool:
    """Signal the whole process group. False once it is gone."""
    try:
        # Children are spawned with start_new_session=True, which makes each
        # one the leader of its own group — so the group id is the pid.
        os.killpg(pid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Kill a process *and its descendants*, then wait, leaving no zombie.

    Killing only the child is not enough, and not just because grandchildren
    are leaks in their own right: `claude` spawns node workers, and a
    grandchild inherits the stdout pipe, so asyncio's `wait()` does not return
    while one is alive — the reap would hang and the slot would stay taken.
    """
    if proc.returncode is not None:
        return

    if not _signal_group(proc.pid, signal.SIGTERM):
        return

    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_TERM_GRACE)
        return
    except asyncio.TimeoutError:
        pass

    if not _signal_group(proc.pid, signal.SIGKILL):
        return

    # SIGKILL cannot be blocked, but the wait still has to happen or the child
    # stays a zombie — which occupies a process slot exactly like a live one.
    # Shielded so an outer cancellation cannot abandon it half-reaped.
    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=5)
    except asyncio.TimeoutError:
        log.warning(f"Process group {proc.pid} survived SIGKILL")


async def run_capture(
    *args: str,
    timeout: float = DEFAULT_TIMEOUT,
    input: bytes | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int | None, bytes, bytes]:
    """Run a command to completion and return (returncode, stdout, stderr).

    Raises asyncio.TimeoutError if the command overruns `timeout` — the same
    exception the old inline `wait_for` raised, so existing handlers still
    work. Unlike the old code, the child is dead by the time it propagates.

    `env=None` (the default) inherits this process's full environment, same
    as before this parameter existed. Pass an explicit dict to override it.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if input is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        # Own process group, so a timeout can take the whole tree down.
        start_new_session=True,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input), timeout=timeout
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        # Cancellation matters as much as timeout here: a dropped WebSocket
        # cancels whatever it was waiting on, and that used to strand the
        # child just as permanently.
        await _reap(proc)
        raise
    except BaseException:
        await _reap(proc)
        raise

    return proc.returncode, stdout, stderr


async def run_detached(*args: str, timeout: float = DEFAULT_TIMEOUT) -> None:
    """Run a command for its side effect only, discarding its output.

    Still waits for it. Fire-and-forget `create_subprocess_exec` calls left a
    zombie plus two pipes behind every time — the commands this is used for
    (`osascript` asking Terminal to open a window) return immediately anyway.
    """
    try:
        await run_capture(*args, timeout=timeout)
    except asyncio.TimeoutError:
        log.warning(f"Detached command timed out: {args[0]}")
    except Exception as e:
        log.warning(f"Detached command failed: {args[0]}: {e}")
