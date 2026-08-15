"""JARVIS battery awareness — power state, and a word before it runs out.

Two jobs. `read()` answers "how much battery do I have left", which JARVIS
could not do at all before. `BatteryMonitor` watches for the machine getting
low and says so once, unprompted, which is the half that actually earns its
keep: the laptop dying mid-sentence is the one failure a butler should have
seen coming.

Everything comes from `pmset -g batt`, which is present on every Mac and needs
no permission prompt, no dependency and no entitlement.

The alert rules are deliberately narrow. Each threshold speaks at most once per
discharge, crossing downward only, and plugging in re-arms them. A monitor that
announces every 3 seconds while the number sits at 20% is not information, it
is an alarm clock you cannot turn off — which is exactly what the naive version
of this does.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass

from proc import run_capture

log = logging.getLogger("jarvis.battery")

# Descending, and read as "at or below". Order matters: checking 20 before 5
# and stopping at the first match is what makes the low thresholds reachable.
ALERT_THRESHOLDS = (20, 10, 5)

# Re-arm a threshold only once the charge is meaningfully back above it, so a
# reading oscillating across the line does not speak twice.
REARM_MARGIN = 3

POLL_SECONDS = 60.0


@dataclass
class BatteryState:
    percent: int
    plugged: bool
    charging: bool
    minutes_remaining: int | None  # None when macOS is still calculating

    @property
    def is_desktop(self) -> bool:
        """No battery at all — an iMac or a Mac mini."""
        return self.percent < 0


# " -InternalBattery-0 (id=23461987)\t37%; discharging; 17:30 remaining present: true"
_PERCENT_RE = re.compile(r"(\d{1,3})%")
_REMAINING_RE = re.compile(r"(\d+):(\d{2})\s+remaining")


def _parse(output: str) -> BatteryState | None:
    """Read `pmset -g batt` output. None when it says nothing useful."""
    if not output.strip():
        return None

    percent_match = _PERCENT_RE.search(output)
    if not percent_match:
        # Desktops print "Now drawing from 'AC Power'" and no battery line.
        return None
    percent = int(percent_match.group(1))

    lowered = output.lower()
    plugged = "ac power" in lowered
    # 'charged' means plugged in and full; 'charging' means actively filling.
    charging = "; charging" in lowered

    minutes = None
    remaining_match = _REMAINING_RE.search(output)
    if remaining_match:
        hours, mins = int(remaining_match.group(1)), int(remaining_match.group(2))
        # macOS prints "0:00 remaining" while it is still estimating; that is
        # not a prediction that the machine dies this instant.
        total = hours * 60 + mins
        minutes = total if total > 0 else None

    return BatteryState(
        percent=percent, plugged=plugged, charging=charging, minutes_remaining=minutes
    )


async def read() -> BatteryState | None:
    """Current power state, or None on a desktop or if pmset misbehaves."""
    try:
        code, stdout, _ = await run_capture("pmset", "-g", "batt", timeout=5)
    except asyncio.TimeoutError:
        log.warning("pmset timed out")
        return None
    except Exception as e:
        log.warning(f"pmset failed: {e}")
        return None

    if code != 0:
        return None
    return _parse(stdout.decode("utf-8", errors="replace"))


class BatteryMonitor:
    """Polls the battery and calls `on_low(state, threshold)` when it drops.

    Like TimerService, it does not know how to speak — the caller injects that.
    """

    def __init__(self, on_low, has_listener=None):
        self._on_low = on_low
        self._has_listener = has_listener
        self._spoken: set[int] = set()
        self._task: asyncio.Task | None = None
        self._last_percent: int | None = None

    def _listening(self) -> bool:
        if self._has_listener is None:
            return True
        try:
            return bool(self._has_listener())
        except Exception:
            return True

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
            log.info("Battery monitor started")

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()

    def _thresholds_to_speak(self, state: BatteryState, previous: int | None) -> int | None:
        """The one threshold this reading should announce, if any.

        `previous` is the last percentage seen, or None on the very first
        reading of the process.
        """
        if state.plugged:
            # On power: nothing to warn about, and everything re-arms.
            self._spoken.clear()
            return None

        # Re-arm anything the charge has climbed back above, which happens
        # after a short stint on the charger that never reached full.
        for level in list(self._spoken):
            if state.percent > level + REARM_MARGIN:
                self._spoken.discard(level)

        crossed = [level for level in ALERT_THRESHOLDS if state.percent <= level]
        if not crossed:
            return None

        # The very first reading of the process arms the monitor rather than
        # firing it. Starting JARVIS on a laptop already at 15% should not open
        # with a warning, and without this every restart during a low-battery
        # afternoon would repeat one — `_spoken` does not survive the process.
        # The charge is in the prompt context regardless, so asking still works.
        if previous is None:
            self._spoken.update(crossed)
            return None

        fresh = [level for level in crossed if level not in self._spoken]

        # Every crossed level counts as delivered, including ones skipped past.
        # A drop from 25% straight to 8% between two polls should say "10", not
        # say "20" now and "10" a minute later.
        self._spoken.update(crossed)

        return min(fresh) if fresh else None

    async def _run(self):
        while True:
            try:
                state = await read()
                # Keep the cached reading fresh for the prompt regardless, but
                # do not burn a threshold while nobody is connected — marking
                # it spoken would lose the warning entirely, and the charge is
                # only going one way.
                if state is not None:
                    # The cached reading feeds the prompt and is always worth
                    # refreshing. Threshold bookkeeping is not: burning one
                    # while nobody is connected would mark the warning
                    # delivered when it was never heard, and the charge only
                    # goes one way, so it would never come round again.
                    note_state(state)
                    # Tracked on every poll, listener or not: this is simply
                    # what the battery was last seen at, and crossing detection
                    # is wrong without it. Only the announcing is gated.
                    previous = self._last_percent
                    self._last_percent = state.percent
                    if self._listening():
                        level = self._thresholds_to_speak(state, previous)
                        if level is not None:
                            log.info(f"Battery at {state.percent}% — announcing {level}% threshold")
                            try:
                                await self._on_low(state, level)
                            except Exception as e:
                                log.error(f"Battery alert could not be announced: {e}")
                await asyncio.sleep(POLL_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"Battery monitor error: {e}")
                await asyncio.sleep(POLL_SECONDS)


# Cached for the system prompt, refreshed by the monitor's own polling so the
# prompt build never waits on a subprocess.
_last_state: BatteryState | None = None
_last_read_at: float = 0.0


def note_state(state: BatteryState | None):
    global _last_state, _last_read_at
    if state is not None:
        _last_state = state
        _last_read_at = time.time()


def format_for_prompt() -> str:
    """Power state as one line of LLM context, or empty on a desktop."""
    if _last_state is None or time.time() - _last_read_at > 600:
        return ""
    state = _last_state
    if state.plugged:
        status = "charging" if state.charging else "on AC power, full"
    else:
        status = "on battery"
    line = f"POWER: {state.percent}% ({status})"
    if state.minutes_remaining and not state.plugged:
        line += f", about {state.minutes_remaining} minutes left"
    return line
