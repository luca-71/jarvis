"""Read today's schedule from Apple Calendar.

Converted from [ACTION:CALENDAR], for the reason in browse.py: the tag is
invisible to the fallback provider.

The tag version dispatched this into the background through
_lookup_and_report, because reading Calendar over AppleScript is slow and the
old flow had already spoken by the time the answer arrived. A tool inverts
that: the answer comes back before JARVIS says anything, so he can actually
tell the user what is on the calendar instead of promising to look. The cost is
that the turn waits — hence the cache below, and a timeout that is generous
rather than optimistic.
"""

import logging

from calendar_access import (
    get_todays_events,
    format_schedule_summary,
    refresh_cache,
)

log = logging.getLogger("jarvis.plugins.calendar")

PLUGIN = {
    "name": "calendar",
    "description": (
        "Today's schedule from Apple Calendar. For just the next event and "
        "minutes remaining, `next_meeting` is faster."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "refresh": {"type": "boolean",
                        "description": "Re-read from Calendar.app (slower); use if they just changed it."},
        },
        "required": [],
    },
    # 45s, and it is the cold start that needs it, not the read. Measured:
    # once Calendar.app is running, get_todays_events() takes 0.4s cold and
    # 0.00s from cache. But the very first call after a machine reboot has to
    # launch Calendar.app and let it sync — ten calendars here — and that blew
    # straight through a 30s timeout. Every call after it is instant.
    "timeout": 45.0,
}


async def run(params: dict) -> str:
    if params.get("refresh"):
        try:
            await refresh_cache()
        except Exception as e:
            log.warning(f"Calendar refresh failed: {e}")

    try:
        events = await get_todays_events()
    except Exception as e:
        log.warning(f"Calendar read failed: {e}")
        return "I could not reach the calendar."

    summary = format_schedule_summary(events)
    log.info(f"Calendar read: {len(events)} events today")
    return summary or "Nothing on the calendar today."
