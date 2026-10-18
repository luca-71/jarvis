"""Warn before a calendar event starts, and answer "how long have I got?".

The proactive half is the point: a meeting you are told about after it started
is not information. check() runs on a cadence and returns a Nudge only inside
the warning window — outside it, nothing, which is what keeps this quiet all
day.

Deliberately narrower than [ACTION:CALENDAR], which reads the whole schedule.
This one is about the next thing and the minutes remaining.
"""

import logging
import os
from datetime import datetime

from calendar_access import get_upcoming_events

log = logging.getLogger("jarvis.plugins.next_meeting")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Warn this far ahead. Ten minutes is enough to finish a paragraph and walk to
# a desk, and short enough that the warning is still about *now*.
WARN_MINUTES = _env_int("JARVIS_MEETING_WARN_MINUTES", 10)

PLUGIN = {
    "name": "next_meeting",
    "description": (
        "The next calendar event and minutes until it starts. For the whole "
        "day's schedule use `calendar`."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
    "timeout": 20.0,
    # Calendar reads go through AppleScript, so this is not free. Two minutes
    # is fine for a ten-minute warning window.
    "check_interval": 120.0,
}


async def _next_event() -> tuple[dict, int] | None:
    """The next timed event today and the whole minutes until it starts."""
    events = await get_upcoming_events(hours=4)
    if not events:
        return None
    # get_upcoming_events already filters out all-day events and anything in
    # the past, but it does not promise an order.
    event = min(events, key=lambda e: e["start_dt"])
    minutes = int((event["start_dt"] - datetime.now()).total_seconds() // 60)
    return event, minutes


async def run(params: dict) -> str:
    try:
        found = await _next_event()
    except Exception as e:
        log.warning(f"Calendar read failed: {e}")
        return "I could not read the calendar."

    if not found:
        return "Nothing else on the calendar for the next few hours."

    event, minutes = found
    title = event.get("title") or "an untitled event"
    if minutes <= 0:
        return f"{title} is starting now."
    if minutes < 60:
        return f"{title} in {minutes} minutes, at {event.get('start', '')}.".replace(" , ", " ")
    hours = minutes // 60
    rest = minutes % 60
    when = f"{hours} hour{'s' if hours > 1 else ''}"
    if rest:
        when += f" and {rest} minutes"
    return f"{title} in {when}, at {event.get('start', '')}."


async def check():
    """Return a Nudge only inside the warning window — otherwise nothing.

    Imported lazily so the plugin stays loadable on its own, which is what the
    tests do.
    """
    from proactive import Nudge

    try:
        found = await _next_event()
    except Exception as e:
        log.debug(f"Calendar check failed: {e}")
        return None

    if not found:
        return None

    event, minutes = found
    if minutes < 0 or minutes > WARN_MINUTES:
        return None

    title = event.get("title") or "an event"
    # The key is the occasion, not the sentence: the countdown changes on every
    # check, and keying on the words would announce the same meeting twice.
    key = f"meeting:{title}:{event['start_dt'].isoformat(timespec='minutes')}"

    if minutes <= 1:
        text = f"{title} is starting now, sir."
    else:
        text = f"{title} in {minutes} minutes, sir."

    # Urgent: a meeting warning that waits for a cooldown window has missed
    # the meeting. This is exactly the case the urgent flag exists for.
    return Nudge(text=text, key=key, urgent=True)
