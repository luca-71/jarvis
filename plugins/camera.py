"""Look through a fixed camera in the house and describe what it shows.

One tool, two kinds of hardware behind it. The user says "guarda l'ingresso";
which brand answers is an implementation detail, and exposing it as two tools
would only give the model a way to pick the wrong one.

  ezviz   RTSP over the LAN when the camera exposes it, otherwise a capture
          through the EZVIZ Open Platform. See ezviz_access.py.
  esp32   A plain HTTP GET to an ESP32-CAM on the local network. No account,
          no cloud, no firmware deciding what to expose — and the image never
          leaves the house.

Configuration, both optional and combinable:

    EZVIZ_CAMERAS=ingresso:L12542160:192.168.1.4
    ESP32_CAMERAS=garage:192.168.1.30,cucina:192.168.1.31/capture

An ESP32 entry is `name:host[:port][/path]`. Path defaults to /capture, which
is what the stock Arduino CameraWebServer sketch serves; some firmwares use
/jpg or /snapshot instead.

The frame is deleted in a finally block on every path. These cameras point at
rooms in the user's home — a still has no business staying on disk because a
function returned early.
"""

import asyncio
import base64
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

import ezviz_access
from proc import run_capture

log = logging.getLogger("jarvis.plugins.camera")

_ESP32_RAW = os.getenv("ESP32_CAMERAS", "").strip()
_ESP32_DEFAULT_PATH = "/capture"


@dataclass(frozen=True)
class Source:
    name: str
    kind: str      # "ezviz" | "esp32"
    url: str = ""  # esp32 only: the full snapshot URL


def _esp32_sources() -> dict[str, Source]:
    found: dict[str, Source] = {}
    for entry in _ESP32_RAW.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, target = entry.partition(":")
        name, target = name.strip().lower(), target.strip()
        if not name or not target:
            log.warning(f"Ignoring malformed ESP32_CAMERAS entry: {entry!r}")
            continue
        if not target.startswith(("http://", "https://")):
            target = f"http://{target}"
        # A bare host gets the stock sketch's snapshot path; anything with a
        # path already is taken as written.
        if "/" not in target.split("://", 1)[1]:
            target += _ESP32_DEFAULT_PATH
        found[name] = Source(name=name, kind="esp32", url=target)
    return found


def _sources() -> dict[str, Source]:
    """Every configured camera by name, both kinds together."""
    sources = {
        name: Source(name=name, kind="ezviz")
        for name in ezviz_access.cameras()
    }
    # ESP32 entries win a name collision: they are the ones the user just
    # configured by hand, and a silent merge would be worse than a clear rule.
    sources.update(_esp32_sources())
    return sources


def _camera_property() -> dict:
    """The `camera` argument, built from what is actually configured.

    An enum of the real names means the model picks "ingresso" or "salotto"
    rather than inventing a room, and the API rejects anything else before the
    plugin runs. With one camera the argument is pointless, so it is omitted.
    """
    names = sorted(_sources())
    if len(names) < 2:
        return {}
    return {
        "camera": {"type": "string", "enum": names,
                   "description": "Which camera to look through."}
    }


PLUGIN = {
    "name": "camera",
    "description": (
        "Look through a fixed security camera in the house and describe what "
        "it shows — who is there, whether anyone is at the door. Not the Mac's "
        "webcam (`webcam`), not the monitor (`screen`)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            **_camera_property(),
            "question": {"type": "string",
                         "description": "What to look for. Omit for a general description."},
        },
        "required": [],
    },
    "timeout": 60.0,
}

_VISION_SYSTEM = (
    "You are JARVIS, looking through the user's home security camera. Describe "
    "what you see concisely and factually: people, movement, notable objects, "
    "the room. 2-3 sentences, no markdown. If the frame is black, empty or "
    "unreadable, say so plainly rather than guessing."
)


# --- ESP32-CAM ---------------------------------------------------------------

async def _grab_esp32(path: Path, source: Source) -> str | None:
    """One JPEG over HTTP. Returns an error string, or None on success."""
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            response = await client.get(source.url)
    except httpx.ConnectError:
        return f"The {source.name} camera is not answering on the network."
    except httpx.TimeoutException:
        return f"The {source.name} camera timed out."
    except Exception as e:
        log.warning(f"ESP32 fetch failed for {source.name}: {e}")
        return f"I could not reach the {source.name} camera."

    if response.status_code != 200:
        return f"The {source.name} camera returned an error ({response.status_code})."

    body = response.content
    # The stock sketch answers a wrong path with an HTML error page and a 200,
    # which would otherwise reach the vision model as a "photograph".
    if not body[:3] == b"\xff\xd8\xff":
        return (f"The {source.name} camera did not return a photo — check the "
                f"snapshot path in ESP32_CAMERAS.")

    path.write_bytes(body)
    return None


# --- EZVIZ -------------------------------------------------------------------

async def _grab_ezviz_local(path: Path, name: str) -> str | None:
    if not shutil.which("ffmpeg"):
        return "ffmpeg is not installed, so I cannot read the camera stream."
    try:
        stream = await ezviz_access.stream_url(prefer_local=True, name=name)
    except ezviz_access.EzvizError as e:
        return f"The camera is not configured: {e}"

    if not stream.local:
        return None  # Not an error — tells the caller to try the cloud route.

    try:
        returncode, _, stderr = await run_capture(
            "ffmpeg",
            # UDP is the default and drops packets on busy WiFi, which shows up
            # as a torn or half-green frame rather than as a failure.
            "-rtsp_transport", "tcp",
            "-i", stream.url,
            "-frames:v", "1",
            "-y", str(path),
            timeout=35,
        )
    except asyncio.TimeoutError:
        return "The camera did not respond in time."

    if returncode != 0:
        detail = stderr.decode(errors="replace")
        if "401" in detail or "Unauthorized" in detail:
            return "The camera refused the credentials — check EZVIZ_VERIFY_CODE."
        if "Connection refused" in detail or "No route to host" in detail:
            return "The camera is not reachable on the network."
        log.warning(f"RTSP capture failed: {detail[-200:]}")
        return "I could not get a picture from the camera."
    return None


async def _grab_ezviz_cloud(path: Path, name: str) -> str | None:
    try:
        url = await ezviz_access.snapshot_url(name=name)
    except ezviz_access.EzvizError as e:
        return f"The camera cloud route is not available: {e}"
    except Exception as e:
        log.warning(f"EZVIZ capture request failed: {e}")
        return "The camera did not return a picture."

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            response = await client.get(url)
        if response.status_code != 200:
            return "The camera image could not be downloaded."
        path.write_bytes(response.content)
    except Exception as e:
        log.warning(f"Snapshot download failed: {e}")
        return "The camera image could not be downloaded."
    return None if path.stat().st_size else "The camera returned an empty image."


async def _grab(path: Path, source: Source) -> str | None:
    if source.kind == "esp32":
        return await _grab_esp32(path, source)

    error = await _grab_ezviz_local(path, source.name)
    if error:
        return error
    # _grab_ezviz_local returns None both on success and when the local route
    # is not configured; the file tells the two apart.
    if not path.stat().st_size:
        return await _grab_ezviz_cloud(path, source.name)
    return None


async def run(params: dict, client=None) -> str:
    sources = _sources()
    if not sources:
        return ("No camera is set up yet — none are configured in my settings.")
    if client is None:
        return "My vision system is not available — no API client is configured."

    question = (params.get("question") or "").strip()
    which = (params.get("camera") or "").strip().lower()

    if which:
        source = sources.get(which)
        if source is None:
            return (f"There is no camera called '{which}' — I have: "
                    f"{', '.join(sorted(sources))}.")
    elif len(sources) == 1:
        source = next(iter(sources.values()))
    else:
        return f"Which camera? I have: {', '.join(sorted(sources))}."

    fd, tmp_name = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        error = await _grab(tmp_path, source)
        if error:
            return error
        if not tmp_path.stat().st_size:
            return f"The {source.name} camera returned an empty frame."

        image_b64 = base64.b64encode(tmp_path.read_bytes()).decode()
        log.info(f"Camera '{source.name}' ({source.kind}) frame captured "
                 f"({tmp_path.stat().st_size} bytes)")

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=_VISION_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "text", "text": question or "What do you see?"},
                ],
            }],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text or "I got a picture but could not make anything of it."

    except Exception as e:
        log.error(f"Camera plugin failed: {e}")
        return "Something went wrong looking through the camera."
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as e:
            log.warning(f"Could not delete camera frame {tmp_path}: {e}")
