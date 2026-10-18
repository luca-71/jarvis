"""Report how the machine itself is doing: CPU, memory, disk, screen brightness.

Battery already reaches the prompt through battery.format_for_prompt(), so it
is deliberately not repeated here — this covers what JARVIS could not see.

Brightness is read-only on purpose. Setting it needs either a third-party
binary or Accessibility-driven key codes, and the key codes were measured
doing nothing at all on this machine: the reading never moved off 32768. A
tool that silently fails is worse than one that does not exist, because the
model reports success either way.
"""

import asyncio
import logging
import re
import shutil

import psutil

from proc import run_capture

log = logging.getLogger("jarvis.plugins.system_status")

PLUGIN = {
    "name": "system_status",
    "description": (
        "Machine health: CPU, memory, free disk, brightness. Reads the "
        "hardware — says nothing about what is on screen (`screen`)."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
    "timeout": 15.0,
}

_BRIGHTNESS_RE = re.compile(
    r'"brightness"=\{"min"=0,"max"=(\d+),"value"=(\d+)'
)


async def _brightness_percent() -> int | None:
    """Screen brightness 0-100, or None when the display does not report it
    (an external monitor generally will not)."""
    try:
        returncode, stdout, _ = await run_capture("ioreg", "-l", timeout=10)
        if returncode != 0:
            return None
        match = _BRIGHTNESS_RE.search(stdout.decode(errors="replace"))
        if not match:
            return None
        maximum, value = int(match.group(1)), int(match.group(2))
        return round(value * 100 / maximum) if maximum else None
    except (asyncio.TimeoutError, Exception):
        return None


def _sync_stats() -> dict:
    """psutil calls are blocking; the loader runs this off the event loop."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = shutil.disk_usage("/")
    return {
        # A real 300ms sample rather than interval=None. The None form diffs
        # against whenever psutil was last called in this process — which for a
        # plugin invoked once an hour means averaging the last hour, and in a
        # standalone test means a flat 0.0. It costs 300ms in a worker thread.
        "cpu": psutil.cpu_percent(interval=0.3),
        "mem_used_pct": memory.percent,
        "mem_available_gb": memory.available / 1e9,
        "swap_used_gb": swap.used / 1e9,
        "disk_free_gb": disk.free / 1e9,
        "disk_used_pct": disk.used * 100 / disk.total,
    }


async def run(params: dict) -> str:
    stats = await asyncio.to_thread(_sync_stats)
    brightness = await _brightness_percent()

    parts = [
        f"CPU {stats['cpu']:.0f}%",
        f"memory {stats['mem_used_pct']:.0f}% used "
        f"({stats['mem_available_gb']:.1f} GB free)",
        f"disk {stats['disk_free_gb']:.0f} GB free "
        f"({stats['disk_used_pct']:.0f}% used)",
    ]

    # Swap in use on an 8 GB machine is the difference between "busy" and
    # "thrashing", and it is the reading that explains a slow Mac when CPU and
    # memory percentages both look unremarkable.
    if stats["swap_used_gb"] >= 1:
        parts.append(f"swap {stats['swap_used_gb']:.1f} GB in use — memory is tight")

    if brightness is not None:
        parts.append(f"brightness {brightness}%")

    return ", ".join(parts) + "."
