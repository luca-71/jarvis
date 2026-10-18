"""Unprompted speech: JARVIS saying something nobody asked for.

The design decision that matters is what this does NOT do. The obvious
implementation — wake every twenty minutes, hand the model the current context,
speak whatever it produces — generates chat. Chat with nothing to report is
noise, and noise from a voice assistant gets the voice assistant turned off.

So nothing here generates anything. A plugin's check() returns a Nudge only
when it has a fact worth interrupting for ("meeting in ten minutes"), and this
service decides whether now is a decent moment to say it. No fact, no speech —
however long the silence.

The gates, in order:

  no listener      nobody connected; talking to an empty room
  recent activity  the user spoke seconds ago; they are mid-conversation
  quiet hours      it is the middle of the night AND they are not around
  cooldown         one interruption per window, urgent ones excepted
  duplicate        this exact thing was already said recently

Everything is configurable and the whole service is one env var away from off.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger("jarvis.proactive")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


ENABLED = os.getenv("JARVIS_PROACTIVE", "1").lower() not in ("0", "false", "no")

# Between the user's last word and anything unprompted. They may simply be
# thinking, and being talked over mid-thought is the fastest way to make this
# feature unwelcome.
MIN_SILENCE = _env_int("JARVIS_PROACTIVE_MIN_SILENCE", 60)

# Between one unprompted remark and the next. Urgent nudges ignore it.
COOLDOWN = _env_int("JARVIS_PROACTIVE_COOLDOWN", 900)

# The same nudge key will not be repeated within this window.
DEDUP_WINDOW = _env_int("JARVIS_PROACTIVE_DEDUP", 21600)  # 6 hours

QUIET_START = _env_int("JARVIS_QUIET_START", 22)  # 22:00
QUIET_END = _env_int("JARVIS_QUIET_END", 8)       # 08:00

# Someone active in the last half hour at 23:30 is awake and working, and
# treating them as asleep would be its own kind of wrong.
QUIET_HOURS_GRACE = _env_int("JARVIS_QUIET_GRACE", 1800)

# How often the loop wakes to consider running checks. Each plugin's own
# check_interval throttles it further; this is just the heartbeat.
TICK = 60


@dataclass
class Nudge:
    """Something worth saying, from a plugin that found a reason to say it.

    `key` is what deduplication works on, so it must identify the *occasion*,
    not the wording: "meeting:standup:2026-08-20T14:00" stays stable across
    five minutes of checks, while the spoken sentence may not.
    """
    text: str
    key: str
    urgent: bool = False


class ProactiveService:
    """Polls plugin watchers and speaks what survives the gates.

    Like TimerService and BatteryMonitor, it does not know how to speak: the
    caller injects `announce`. That keeps quiet mode, TTS and WebSocket
    fan-out in the one place that already handles them.
    """

    def __init__(self, registry, announce, has_listener=None,
                 last_activity=None, client_provider=None):
        self._registry = registry
        self._announce = announce
        self._has_listener = has_listener
        self._last_activity = last_activity      # () -> unix ts of last user speech
        self._client_provider = client_provider  # () -> Anthropic client or None
        self._task: asyncio.Task | None = None
        self._last_spoke = 0.0
        self._last_check: dict[str, float] = {}  # plugin name -> ts
        self._said: dict[str, float] = {}        # nudge key -> ts

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        if not ENABLED:
            log.info("Proactive service disabled (JARVIS_PROACTIVE=0)")
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
            log.info(
                f"Proactive service started "
                f"(silence {MIN_SILENCE}s, cooldown {COOLDOWN}s, "
                f"quiet {QUIET_START}:00-{QUIET_END}:00)"
            )

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()

    # -- gates --------------------------------------------------------------

    def _listening(self) -> bool:
        if self._has_listener is None:
            return True
        try:
            return bool(self._has_listener())
        except Exception:
            return False

    def _seconds_since_activity(self) -> float:
        if self._last_activity is None:
            return float("inf")
        try:
            return time.time() - float(self._last_activity())
        except Exception:
            return float("inf")

    def _in_quiet_hours(self, idle_seconds: float) -> bool:
        """Night, and the user is not around to be interrupted anyway.

        Someone who spoke a minute ago at 02:00 is awake; the point of quiet
        hours is not to enforce a bedtime, it is to avoid a voice coming out of
        a dark room.
        """
        if idle_seconds < QUIET_HOURS_GRACE:
            return False
        hour = datetime.now().hour
        if QUIET_START == QUIET_END:
            return False
        if QUIET_START < QUIET_END:
            return QUIET_START <= hour < QUIET_END
        # Wraps past midnight, the usual case (22 -> 8).
        return hour >= QUIET_START or hour < QUIET_END

    def _recently_said(self, key: str) -> bool:
        said_at = self._said.get(key)
        return said_at is not None and (time.time() - said_at) < DEDUP_WINDOW

    def _forget_old(self):
        cutoff = time.time() - DEDUP_WINDOW
        self._said = {k: t for k, t in self._said.items() if t >= cutoff}

    # -- loop ---------------------------------------------------------------

    async def _run(self):
        # Never fire in the first minute of a process: a restart is not an
        # occasion, and the user has just watched the thing come up.
        await asyncio.sleep(TICK)
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"Proactive tick failed: {e}")
            await asyncio.sleep(TICK)

    async def _tick(self):
        if not self._listening():
            return

        idle = self._seconds_since_activity()
        if idle < MIN_SILENCE or self._in_quiet_hours(idle):
            return

        watchers = self._registry.watchers() if self._registry else []
        if not watchers:
            return

        now = time.time()
        client = self._client_provider() if self._client_provider else None

        for rec in watchers:
            # Each watcher keeps its own cadence — several of these reach for
            # AppleScript and should not run every heartbeat.
            if now - self._last_check.get(rec.name, 0) < rec.check_interval:
                continue
            self._last_check[rec.name] = now

            # Isolated per watcher, not just per tick: run_check already
            # swallows what a plugin's check() throws, but anything that gets
            # past it would otherwise silence every watcher after this one in
            # the loop — one broken plugin taking the rest down with it.
            try:
                result = await self._registry.run_check(rec, client=client)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"Watcher '{rec.name}' failed: {e}")
                continue

            nudge = _coerce(result, rec.name)
            if nudge is None:
                continue

            if self._recently_said(nudge.key):
                continue
            if not nudge.urgent and (now - self._last_spoke) < COOLDOWN:
                # Held rather than dropped silently: this is the gate most
                # likely to be blamed for a missing announcement.
                log.debug(f"Nudge from {rec.name} held by cooldown: {nudge.key}")
                continue

            await self._speak(nudge, rec.name)
            # One interruption per tick. Two things at once is a briefing
            # nobody asked for.
            return

    async def _speak(self, nudge: Nudge, source: str):
        self._said[nudge.key] = time.time()
        self._last_spoke = time.time()
        self._forget_old()
        log.info(f"Proactive ({source}): {nudge.text}")
        await self._announce(nudge.text)


def _coerce(result, source: str) -> Nudge | None:
    """Accept a Nudge, or a bare string for the simple case.

    A string gets its text as the dedup key, which is right for a watcher whose
    message only changes when the situation does.
    """
    if result is None:
        return None
    if isinstance(result, Nudge):
        return result if result.text.strip() else None
    if isinstance(result, str):
        text = result.strip()
        return Nudge(text=text, key=f"{source}:{text}") if text else None
    log.warning(f"Plugin '{source}' check returned {type(result).__name__}, ignored")
    return None
