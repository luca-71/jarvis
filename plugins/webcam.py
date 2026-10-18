"""Look through the webcam and describe what is there.

The one capability JARVIS genuinely lacked: he could photograph the monitor
(screen.take_screenshot) but not the room. Same vision path as describe_screen,
pointed at the camera instead of the display.

Capture is ffmpeg + avfoundation, which is already installed. `-framerate 30`
is not optional: without it ffmpeg fails to negotiate a format with the
FaceTime camera and dies with "Input/output error" — measured, not guessed.

The frame is written to a temp file, sent, and deleted in a finally block. A
photograph of whoever is sitting at the machine is not something to leave lying
around on disk because a code path returned early.
"""

import asyncio
import base64
import logging
import os
import shutil
import tempfile
from pathlib import Path

from proc import run_capture

log = logging.getLogger("jarvis.plugins.webcam")

# Device 0 is the built-in camera. A Mac with Continuity Camera also lists the
# iPhone, so this is overridable rather than hardcoded.
_DEVICE = os.getenv("JARVIS_CAMERA_INDEX", "0")

PLUGIN = {
    "name": "webcam",
    "description": (
        "Photograph the user with the webcam and describe it — them, what "
        "they hold, the room. Camera points AT the user; for the monitor use "
        "`screen`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string",
                         "description": "What to look for. Omit for a general description."},
        },
        "required": [],
    },
    # Camera warm-up plus a vision call. Generous, because the alternative to
    # waiting is reporting failure on a shot that was about to arrive.
    "timeout": 45.0,
}

_VISION_SYSTEM = (
    "You are JARVIS, looking through the user's webcam. Describe what you see "
    "concisely and factually: who is there, what they appear to be doing, "
    "notable objects, the room. 2-3 sentences, no markdown. If the frame is "
    "black or unreadable, say so plainly rather than guessing."
)


async def _capture(path: Path) -> str | None:
    """Grab one frame. Returns an error string, or None on success."""
    if not shutil.which("ffmpeg"):
        return "ffmpeg is not installed, so I cannot reach the camera."

    try:
        returncode, _, stderr = await run_capture(
            "ffmpeg",
            "-f", "avfoundation",
            "-framerate", "30",
            "-i", _DEVICE,
            "-frames:v", "1",
            "-y", str(path),
            timeout=20,
        )
    except asyncio.TimeoutError:
        return "The camera did not respond."

    if returncode != 0:
        detail = stderr.decode(errors="replace")
        # The distinctive failure when macOS has not granted camera access —
        # worth naming, because the fix is a checkbox and not something the
        # user could guess from "capture failed".
        if "Input/output error" in detail or "Operation not permitted" in detail:
            return ("The camera would not open. JARVIS may need camera "
                    "permission in System Settings under Privacy and Security.")
        log.warning(f"ffmpeg capture failed: {detail[-200:]}")
        return "I could not get a picture from the camera."

    if not path.exists() or path.stat().st_size == 0:
        return "The camera returned an empty frame."
    return None


async def run(params: dict, client=None) -> str:
    if client is None:
        return "My vision system is not available — no API client is configured."

    question = (params.get("question") or "").strip()

    fd, tmp_name = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        error = await _capture(tmp_path)
        if error:
            return error

        image_b64 = base64.b64encode(tmp_path.read_bytes()).decode()
        log.info(f"Webcam frame captured ({tmp_path.stat().st_size} bytes)")

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=_VISION_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": question or "What do you see?"},
                ],
            }],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text or "I got a picture but could not make anything of it."

    except Exception as e:
        log.error(f"Webcam plugin failed: {e}")
        return "Something went wrong looking through the camera."
    finally:
        # Always, on every path — see the module docstring.
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as e:
            log.warning(f"Could not delete webcam frame {tmp_path}: {e}")
