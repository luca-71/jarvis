"""Open any macOS application by name.

The first real plugin, and a genuine gap: JARVIS could open Terminal and
Chrome, because each had its own AppleScript and its own [ACTION:X] tag, but
not Spotify, Xcode or Mail — every new app meant another tag in the prompt.
"""

import asyncio
import logging

from proc import run_capture

log = logging.getLogger("jarvis.plugins.open_app")

PLUGIN = {
    "name": "open_app",
    "description": (
        "Launch a macOS application by name. Not for websites (`browse`) or "
        "a coding Terminal ([ACTION:OPEN_TERMINAL])."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "app_name": {"type": "string", "description": "e.g. 'Spotify', 'Visual Studio Code'"},
        },
        "required": ["app_name"],
    },
    "timeout": 15.0,
}


async def run(params: dict) -> str:
    app = (params.get("app_name") or "").strip()
    if not app:
        return "No application name was given."

    try:
        # `open -a` resolves display names loosely, so "vs code" finds "Visual
        # Studio Code" — which is the whole point when the name arrives from
        # speech recognition rather than from a file listing.
        returncode, _, stderr = await run_capture("open", "-a", app, timeout=10)
    except asyncio.TimeoutError:
        return f"{app} did not respond to the launch request."

    if returncode != 0:
        detail = stderr.decode(errors="replace").strip()
        if "Unable to find application" in detail:
            return f"There is no application named {app} on this machine."
        log.warning(f"open -a {app} failed: {detail[:200]}")
        return f"{app} could not be opened."

    log.info(f"Opened {app}")
    return f"{app} is open."
