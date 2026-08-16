"""JARVIS EZVIZ access — a live camera feed, on request.

Two routes to the same picture, and the choice between them is not stylistic.

The cloud route (EZVIZ Open Platform) hands back a short-lived HLS URL that
plays anywhere, including when the user is not at home. It costs a round trip
to Hangzhou and it rate-limits the token endpoint hard, so the token is cached
and refreshed behind a lock — see `_token()`.

The local route builds an RTSP URL straight to the camera on the LAN. No cloud,
no token, no expiry, and about a second less latency. It is the better answer
whenever it is available, which is why `stream_url()` prefers it when
EZVIZ_CAMERA_HOST is set.

Credentials: the Open Platform wants an appKey/appSecret pair from the
developer console, NOT the account email and password. The account password
authenticates only the mobile app's private API, which is undocumented and
changes without notice. If you find yourself wanting to put an account password
in here, use the RTSP route instead — it needs only the verification code
printed on the camera, which is scoped to that one device.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from urllib.parse import quote

import httpx

log = logging.getLogger("jarvis.ezviz")

APP_KEY = os.getenv("EZVIZ_APP_KEY", "").strip()
APP_SECRET = os.getenv("EZVIZ_APP_SECRET", "").strip()

# Serials are 9 characters and the API is case-sensitive — a lowercase serial
# comes back as "device does not exist", which reads like the wrong camera
# rather than the wrong case.
DEVICE_SERIAL = os.getenv("EZVIZ_DEVICE_SERIAL", "").strip().upper()

# Set when the camera's stream is encrypted (most are, by default). Also the
# RTSP password — EZVIZ uses the verification code for both.
VERIFY_CODE = os.getenv("EZVIZ_VERIFY_CODE", "").strip()

# LAN address of the camera. Set it and the local RTSP route wins.
CAMERA_HOST = os.getenv("EZVIZ_CAMERA_HOST", "").strip()
CAMERA_RTSP_PORT = os.getenv("EZVIZ_RTSP_PORT", "554").strip() or "554"

# The token response names the region domain to use for every later call;
# this is only the bootstrap host. Europe accounts get isgpopen.ezvizlife.com
# back, so hardcoding the global one for device calls returns 404s.
BOOTSTRAP_HOST = os.getenv("EZVIZ_API_HOST", "https://open.ezvizlife.com").rstrip("/")

_HTTP_TIMEOUT = 10.0

# Tokens are valid 7 days. Refresh an hour early: a token that expires mid-call
# fails the request, and the endpoint is rate-limited enough that retrying on
# 10002 in a loop is a good way to get the app key throttled.
_TOKEN_REFRESH_MARGIN = 3600.0

# Stream URLs expire too, and much faster. Ask for the longest the API allows
# and re-request rather than caching near the edge.
_STREAM_EXPIRE_SECONDS = 3600


def configured() -> bool:
    """True when either route has what it needs. Everything else no-ops."""
    if not DEVICE_SERIAL:
        return False
    return bool((CAMERA_HOST and VERIFY_CODE) or (APP_KEY and APP_SECRET))


@dataclass
class Stream:
    url: str
    protocol: str  # "rtsp" | "hls" | "rtmp" | "flv"
    local: bool
    expires_at: float | None  # None for RTSP, which does not expire


# --- Cloud route ------------------------------------------------------------

_access_token: str | None = None
_token_expires_at: float = 0.0
_area_domain: str = BOOTSTRAP_HOST

# Serialises refreshes. Without it, three concurrent "show me the camera"
# requests fire three token calls, and EZVIZ counts every one against the
# quota even though two are redundant.
_token_lock = asyncio.Lock()


class EzvizError(RuntimeError):
    """An API call came back with a non-200 code."""


def _check(payload: dict, what: str) -> dict:
    """Unwrap an EZVIZ response body, raising on its in-band error codes.

    The HTTP status is 200 even for failures; the real result is in `code`,
    as a string.
    """
    code = str(payload.get("code", ""))
    if code == "200":
        return payload.get("data") or {}

    msg = payload.get("msg", "no message")
    # The codes worth naming, because the generic message does not explain them.
    hint = {
        "10001": "parameter error — check the device serial",
        "10002": "access token expired or invalid",
        "10005": "appKey is frozen",
        "20002": "device does not exist on this account",
        "20006": "network error reaching the camera",
        "20007": "camera is offline",
        "20018": "this account does not own that device",
        "60020": "the account lacks permission for this call",
    }.get(code)
    detail = f"{msg} (code {code})" + (f": {hint}" if hint else "")
    raise EzvizError(f"{what} failed — {detail}")


async def _token(force: bool = False) -> str:
    """A valid access token, cached. Refreshes under a lock."""
    global _access_token, _token_expires_at, _area_domain

    if not (APP_KEY and APP_SECRET):
        raise EzvizError("EZVIZ_APP_KEY / EZVIZ_APP_SECRET are not set")

    if not force and _access_token and time.time() < _token_expires_at:
        return _access_token

    async with _token_lock:
        # Re-check inside the lock: whoever held it may have just refreshed.
        if not force and _access_token and time.time() < _token_expires_at:
            return _access_token

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as http:
            resp = await http.post(
                f"{BOOTSTRAP_HOST}/api/lapp/token/get",
                data={"appKey": APP_KEY, "appSecret": APP_SECRET},
            )
        resp.raise_for_status()
        data = _check(resp.json(), "Token request")

        _access_token = data["accessToken"]
        # expireTime is epoch milliseconds.
        _token_expires_at = data["expireTime"] / 1000.0 - _TOKEN_REFRESH_MARGIN
        if data.get("areaDomain"):
            _area_domain = str(data["areaDomain"]).rstrip("/")

        # Never log the token itself.
        log.info(f"EZVIZ token acquired, area domain {_area_domain}")
        return _access_token


async def _cloud_stream(protocol: str = "hls", quality: int = 1) -> Stream:
    """Ask the platform for a playable URL.

    `quality` is 1 for HD and 2 for the fluent (lower-bitrate) substream.
    """
    protocol_codes = {"ezopen": 1, "hls": 2, "rtmp": 3, "flv": 4}
    if protocol not in protocol_codes:
        raise ValueError(f"Unsupported protocol: {protocol}")

    async def request(token: str) -> httpx.Response:
        form = {
            "accessToken": token,
            "deviceSerial": DEVICE_SERIAL,
            "channelNo": "1",
            "protocol": str(protocol_codes[protocol]),
            "quality": str(quality),
            "expireTime": str(_STREAM_EXPIRE_SECONDS),
        }
        # Only sent for encrypted cameras; an empty code on an unencrypted one
        # is rejected as a bad parameter rather than ignored.
        if VERIFY_CODE:
            form["code"] = VERIFY_CODE
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as http:
            return await http.post(
                f"{_area_domain}/api/lapp/v2/live/address/get", data=form
            )

    token = await _token()
    resp = await request(token)
    resp.raise_for_status()
    payload = resp.json()

    # One forced refresh on an expired token, then give up. A token can go
    # stale between the cache check and the call.
    if str(payload.get("code")) == "10002":
        log.info("EZVIZ token rejected as stale — refreshing once")
        resp = await request(await _token(force=True))
        resp.raise_for_status()
        payload = resp.json()

    data = _check(payload, "Live address request")
    return Stream(
        url=data["url"],
        protocol=protocol,
        local=False,
        expires_at=time.time() + _STREAM_EXPIRE_SECONDS,
    )


# --- Local route ------------------------------------------------------------

def _local_stream(quality: int = 1) -> Stream:
    """RTSP straight to the camera. No network round trip, no expiry.

    EZVIZ cameras authenticate RTSP as user `admin` with the verification code
    as the password — the same code printed on the device label. Channel path
    /h264/ch1/main is the HD stream; /sub is the lower-bitrate one.
    """
    if not (CAMERA_HOST and VERIFY_CODE):
        raise EzvizError("EZVIZ_CAMERA_HOST / EZVIZ_VERIFY_CODE are not set")

    # Verification codes are uppercase alphanumeric, but quote anyway — a code
    # containing '@' or '/' would otherwise silently truncate the host.
    password = quote(VERIFY_CODE, safe="")
    substream = "sub" if quality == 2 else "main"
    return Stream(
        url=f"rtsp://admin:{password}@{CAMERA_HOST}:{CAMERA_RTSP_PORT}/h264/ch1/{substream}/av_stream",
        protocol="rtsp",
        local=True,
        expires_at=None,
    )


# --- Public API -------------------------------------------------------------

async def stream_url(prefer_local: bool = True, quality: int = 1) -> Stream:
    """A playable URL for the camera, local route first when available.

    Falls back to the cloud if the local route is not configured. Raises
    EzvizError when neither is.
    """
    if prefer_local and CAMERA_HOST and VERIFY_CODE:
        return _local_stream(quality=quality)
    return await _cloud_stream(quality=quality)


async def snapshot_url() -> str:
    """A still image from the camera — cheaper than a stream for 'is anyone there'."""
    token = await _token()
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as http:
        resp = await http.post(
            f"{_area_domain}/api/lapp/device/capture",
            data={"accessToken": token, "deviceSerial": DEVICE_SERIAL, "channelNo": "1"},
        )
    resp.raise_for_status()
    return _check(resp.json(), "Capture request")["picUrl"]


async def device_status() -> dict:
    """Online state, signal, battery where applicable. {} when unreachable."""
    try:
        token = await _token()
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as http:
            resp = await http.post(
                f"{_area_domain}/api/lapp/device/info",
                data={"accessToken": token, "deviceSerial": DEVICE_SERIAL},
            )
        resp.raise_for_status()
        return _check(resp.json(), "Device info request")
    except Exception as e:
        log.warning(f"EZVIZ device status unavailable: {e}")
        return {}


_last_status: dict = {}
_last_status_at: float = 0.0


def note_status(status: dict) -> None:
    global _last_status, _last_status_at
    if status:
        _last_status = status
        _last_status_at = time.time()


def format_for_prompt() -> str:
    """Camera state as one line of LLM context, empty when unconfigured."""
    if not configured():
        return ""
    if not _last_status or time.time() - _last_status_at > 600:
        return "CAMERA: available (state unknown)"
    name = _last_status.get("deviceName", "the camera")
    online = "online" if _last_status.get("status") == 1 else "offline"
    return f"CAMERA: {name} is {online}"
