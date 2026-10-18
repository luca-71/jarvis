"""Read and set the system output volume.

One tool rather than three (set/mute/report), because the three phrasings a
user actually says — "volume at 30", "turn it down", "mute" — are the same
intent with different arguments, and splitting them into separate tools only
gives the model more ways to pick the wrong one.
"""

import asyncio
import logging

from proc import run_capture

log = logging.getLogger("jarvis.plugins.volume")

PLUGIN = {
    "name": "volume",
    # Descriptions are paid for on every turn, twice per turn when a tool
    # fires — so they are written to the shortest form that still separates
    # this tool from its neighbours, not to the clearest English.
    "description": (
        "System output volume: set, nudge, mute, or report. No arguments "
        "reports the current level. Not per-app volume."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "level": {"type": "integer", "description": "Absolute 0-100.",
                      "minimum": 0, "maximum": 100},
            "relative": {"type": "integer",
                         "description": "Delta, e.g. 10 for 'turn it up'. Ignored if level given.",
                         "minimum": -100, "maximum": 100},
            "muted": {"type": "boolean", "description": "true mutes, false unmutes."},
        },
        "required": [],
    },
    "timeout": 10.0,
}


async def _osascript(script: str) -> str:
    returncode, stdout, stderr = await run_capture("osascript", "-e", script, timeout=5)
    if returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace").strip()[:120])
    return stdout.decode(errors="replace").strip()


async def _state() -> tuple[int, bool]:
    """Current (volume 0-100, muted)."""
    raw = await _osascript(
        "set s to get volume settings\n"
        'return (output volume of s as text) & "|" & (output muted of s as text)'
    )
    level_str, muted_str = raw.split("|", 1)
    return int(level_str), muted_str.strip().lower() == "true"


async def run(params: dict) -> str:
    try:
        level, muted = await _state()
    except asyncio.TimeoutError:
        return "The sound settings did not respond."
    except Exception as e:
        log.warning(f"Could not read volume: {e}")
        return "I could not read the current volume."

    wants_level = params.get("level")
    wants_relative = params.get("relative")
    wants_muted = params.get("muted")

    # Nothing to change — just report.
    if wants_level is None and wants_relative is None and wants_muted is None:
        return f"Volume is at {level}%{', muted' if muted else ''}."

    changes = []
    try:
        if wants_level is not None or wants_relative is not None:
            if wants_level is not None:
                target = int(wants_level)
            else:
                target = level + int(wants_relative)
            # Clamped rather than rejected: "turn it up" at 95 is a reasonable
            # thing to say, and the honest answer is "it is already at maximum",
            # not an error about the argument being out of range.
            target = max(0, min(100, target))
            await _osascript(f"set volume output volume {target}")
            if target == level:
                changes.append(f"already at {target}%")
            else:
                changes.append(f"{target}%")
            level = target

        if wants_muted is not None:
            await _osascript(f"set volume output muted {str(bool(wants_muted)).lower()}")
            # Unmuting at zero volume is silence with extra steps, so say so
            # rather than reporting success into a machine that stays quiet.
            if not wants_muted and level == 0:
                changes.append("unmuted, but the volume is at zero")
            else:
                changes.append("muted" if wants_muted else "unmuted")
    except asyncio.TimeoutError:
        return "The sound settings stopped responding partway through."
    except Exception as e:
        log.warning(f"Could not set volume: {e}")
        return "I could not change the volume."

    log.info(f"Volume: {', '.join(changes)}")
    return "Volume " + " and ".join(changes) + "."
