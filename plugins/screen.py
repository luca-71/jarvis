"""Look at the user's display and describe it.

Converted from [ACTION:SCREEN], same reason as browse and calendar.

`describe_screen` already does the work — screenshot through vision, falling
back to the window list when vision is unavailable — and it takes the model
client as its argument, which the loader injects into any run() that asks for
it. So this plugin is mostly a declaration: the value is in the description
telling the model when NOT to reach for it.

Opposite direction from the webcam plugin, and the pair is the single most
confusable thing in the catalogue: this photographs the monitor, webcam
photographs the person. Both descriptions say so explicitly.
"""

import logging

from screen import describe_screen

log = logging.getLogger("jarvis.plugins.screen")

PLUGIN = {
    "name": "screen",
    "description": (
        "Capture the monitor and describe it — open apps, visible content, "
        "on-screen text. Photographs the DISPLAY: for the calendar use "
        "`calendar`, for the user themselves use `webcam`."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
    # Screenshot plus a vision call.
    "timeout": 45.0,
}


async def run(params: dict, client=None) -> str:
    if client is None:
        return "My vision system is not available — no API client is configured."
    try:
        result = await describe_screen(client)
    except Exception as e:
        log.warning(f"Screen description failed: {e}")
        return "I could not get a look at the screen."
    log.info("Screen described")
    return result or "I could not make anything out on the screen."
