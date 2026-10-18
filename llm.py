"""Anthropic-shaped client with an OpenRouter fallback underneath.

The entire codebase calls `client.messages.create(...)` and reads
`response.content[i].text`, `.stop_reason`, `.usage.input_tokens`. Seventeen
call sites across five files. Rewriting them for a second provider would be a
week of work and a permanent tax on every future change, so this does the
opposite: it presents exactly the surface the code already uses, and swaps what
happens behind it.

When Anthropic returns "your credit balance is too low", the next call goes to
OpenRouter instead. Anthropic is retried after RETRY_AFTER seconds, because the
balance being topped up is the expected end of this state, not a permanent one.

What the fallback deliberately does NOT do:

  - Nothing switches on a 429 or a network blip. Those are retried, not
    escalated; failing over on a transient error would send traffic to the
    second provider for a problem that fixed itself.
  - Prompt caching does not survive the trip. The static/dynamic split in
    generate_response() is built around Anthropic's cache; OpenRouter's models
    work on other rules entirely (if at all). On the fallback path every turn
    pays full price — which is fine, because the fallback path exists for when
    the alternative is not answering at all.

OpenRouter speaks the OpenAI chat-completions dialect, which is a much closer
match to Anthropic's shape than Gemini's was: standard JSON Schema for tools
(no type-casing dance), and tool replies are their own "tool" messages keyed
by call id (no name lookup needed, unlike Gemini's functionResponse-by-name).

UNVERIFIED, unlike the Gemini path this replaced: whether the configured
model tolerates the plugin tool declarations *and* still emits literal
[ACTION:X] tags as text when it decides to act outside the plugin set. The
Gemini fallback measured 0/3 on that; OpenRouter's default model here is a
"stealth" listing — an unannounced model an OpenRouter partner runs in
exchange for eval traffic — so it has not been run through that check. Watch
`jarvis.llm` logs after the first real fallback and adjust if tags come back
silent the way they did for Gemini.

Privacy note worth keeping in view: a "stealth" OpenRouter model is hosted by
whichever lab is trialling it, specifically to collect usage — that is the
deal that makes it free or cheap. JARVIS sends mail, calendar, notes,
screenshots and webcam frames through this client. OPENROUTER_API_KEY being
set is the consent to that trade; nothing routes there unless a key is
present.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("jarvis.llm")

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# OpenRouter's own naming for its author/model-slug.
#
# A pinned model, not "openrouter/free" (its own auto-routed free tier):
# that router picks at random across whatever free models are up, including
# ones with no business answering a conversational turn at all — one draw
# landed on nvidia/nemotron-3.5-content-safety:free, a moderation classifier,
# which duly returned "User Safety: safe" as JARVIS's reply to "buongiorno".
# nvidia/nemotron-3-super-120b-a12b:free measured 8/8 clean Italian replies
# with tools declared, both with and without a tool call actually needed.
#
# Free listings churn: stealth/ox-alpha started returning empty HTTP 200s
# under load, and nvidia/nemotron-nano-12b-v2-vl:free was later retired
# outright, 404ing with no warning. Check https://openrouter.ai/models if
# this one starts erroring too.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

# How long to stay on the fallback before trying Anthropic again.
RETRY_AFTER = int(os.getenv("JARVIS_LLM_RETRY_AFTER", 600))

# Substrings that mean "this key cannot pay for this call" rather than "try
# again in a moment". Matched case-insensitively against the error text.
_OUT_OF_CREDIT = (
    "credit balance is too low",
    "billing",
    "insufficient_quota",
)


# ---------------------------------------------------------------------------
# Anthropic-shaped response objects
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class Response:
    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    model: str = ""


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI chat-completions translation
# ---------------------------------------------------------------------------

def _system_text(system) -> str:
    """Flatten Anthropic's system blocks into one instruction string."""
    if not system:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(p for p in parts if p)


def _to_openrouter(kwargs: dict) -> dict:
    messages: list[dict] = []

    system = _system_text(kwargs.get("system"))
    if system:
        messages.append({"role": "system", "content": system})

    for message in kwargs.get("messages", []):
        role = message.get("role", "user")
        content = message.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        image_parts: list[dict] = []
        tool_calls: list[dict] = []
        # tool_result blocks answer a call made in the PRECEDING assistant
        # message; OpenAI wants each as its own "tool" message rather than
        # bundled into the surrounding turn, so they're collected separately
        # and flushed before whatever text/tool_calls this turn also carries.
        tool_results: list[tuple[str, str]] = []

        for block in content:
            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

            if btype == "text":
                text_parts.append(block.get("text") if isinstance(block, dict) else block.text)

            elif btype == "image":
                source = block.get("source") if isinstance(block, dict) else block.source
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                })

            elif btype == "tool_use":
                if isinstance(block, dict):
                    call_id, name, args = block.get("id"), block.get("name"), block.get("input")
                else:
                    call_id, name, args = block.id, block.name, block.input
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args or {})},
                })

            elif btype == "tool_result":
                if isinstance(block, dict):
                    call_id, result = block.get("tool_use_id", ""), block.get("content")
                else:
                    call_id, result = getattr(block, "tool_use_id", ""), getattr(block, "content", "")
                tool_results.append((call_id, result if isinstance(result, str) else str(result)))

        for call_id, result_text in tool_results:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result_text})

        if text_parts or image_parts or tool_calls:
            out: dict = {"role": role}
            if image_parts:
                out["content"] = [{"type": "text", "text": t} for t in text_parts if t] + image_parts
            elif text_parts:
                out["content"] = "\n".join(t for t in text_parts if t)
            else:
                out["content"] = None
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)

    body: dict = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        # Several free reasoning models write their chain-of-thought straight
        # into `content` — there is no separate `reasoning` field to leave it
        # in — always in English regardless of what the system prompt asks
        # for, before ever reaching the actual answer. `exclude: true` asks
        # OpenRouter to drop that trace; `effort: low` asks the model to
        # spend less of the budget producing it. This alone is NOT enough —
        # see the max_tokens floor below for the other half of the fix.
        "reasoning": {"effort": "low", "exclude": True},
    }

    if kwargs.get("max_tokens"):
        # Reproduced directly against JARVIS's real system prompt (~10K
        # chars: personality, tool docs, action-tag examples): at
        # max_tokens=250 — what a normal chat turn asks for — the model
        # spent the entire budget narrating its plan in English ("Okay, the
        # user said 'buongiorno' which is Italian for... I need to respond
        # in Italian but keep 'Maestro'...") and got cut off before writing
        # a single word of the actual reply. `reasoning: exclude` above
        # doesn't prevent this for a model that has no separate reasoning
        # channel to exclude from — the only lever left is giving the whole
        # turn, narration included, enough room to run its course and still
        # leave tokens for the answer. 1500 cleared every case tried,
        # including the one open-ended prompt that used 949 completion
        # tokens end to end. Like Gemini's equivalent floor above, this
        # doesn't make JARVIS more verbose — the 1-2 sentence limit is the
        # system prompt's job, not this ceiling's.
        body["max_tokens"] = max(kwargs["max_tokens"], 1500)
    if kwargs.get("temperature") is not None:
        body["temperature"] = kwargs["temperature"]

    tools = kwargs.get("tools")
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
            for tool in tools
        ]

    return body


def _from_openrouter(payload: dict, model: str) -> Response:
    choices = payload.get("choices") or []
    blocks: list = []
    stop_reason = "end_turn"

    if choices:
        message = choices[0].get("message", {})

        text = message.get("content")
        if text:
            blocks.append(TextBlock(text=text))

        tool_calls = message.get("tool_calls") or []
        for call in tool_calls:
            func = call.get("function", {})
            try:
                args = json.loads(func.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            blocks.append(ToolUseBlock(id=call.get("id", ""), name=func.get("name", ""), input=args))

        if tool_calls:
            stop_reason = "tool_use"
        else:
            finish = choices[0].get("finish_reason") or "stop"
            stop_reason = "max_tokens" if finish == "length" else "end_turn"

    usage = payload.get("usage") or {}
    return Response(
        content=blocks,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        ),
        model=model,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _is_out_of_credit(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _OUT_OF_CREDIT)


class _Stream:
    """Mimics `client.messages.stream()`: async context manager, .text_stream."""

    def __init__(self, owner, kwargs):
        self._owner = owner
        self._kwargs = kwargs
        self._inner = None
        self._inner_cm = None

    async def __aenter__(self):
        if not self._owner.using_fallback:
            try:
                self._inner_cm = self._owner._anthropic.messages.stream(**self._kwargs)
                self._inner = await self._inner_cm.__aenter__()
                return self._inner
            except Exception as e:
                if not (_is_out_of_credit(e) and self._owner.can_fallback):
                    raise
                self._owner._switch(e)
        self._inner_cm = None
        return self

    async def __aexit__(self, *exc):
        if self._inner_cm is not None:
            return await self._inner_cm.__aexit__(*exc)
        return False

    @property
    def text_stream(self):
        return self._owner._openrouter_text_stream(self._kwargs)


class _Messages:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, **kwargs):
        return await self._owner._create(kwargs)

    def stream(self, **kwargs):
        return _Stream(self._owner, kwargs)


class LLMClient:
    """Drop-in for anthropic.AsyncAnthropic, with OpenRouter behind it."""

    def __init__(self, anthropic_client, openrouter_key: str = ""):
        self._anthropic = anthropic_client
        self._openrouter_key = openrouter_key or ""
        self._fallback_since = 0.0
        self.messages = _Messages(self)

    # -- state --------------------------------------------------------------

    @property
    def can_fallback(self) -> bool:
        return bool(self._openrouter_key)

    @property
    def using_fallback(self) -> bool:
        if not self._fallback_since:
            return False
        if time.time() - self._fallback_since > RETRY_AFTER:
            # The balance may have been topped up. Cheaper to find out with one
            # real call than to keep a second provider forever.
            log.info("Retrying Anthropic after fallback window")
            self._fallback_since = 0.0
            return False
        return True

    def _switch(self, error: Exception):
        if not self._fallback_since:
            log.warning(f"Anthropic unavailable ({str(error)[:120]}) — "
                        f"falling back to {OPENROUTER_MODEL}")
        self._fallback_since = time.time()

    def status(self) -> dict:
        return {
            "provider": "openrouter" if self.using_fallback else "anthropic",
            "fallback_available": self.can_fallback,
            "fallback_model": OPENROUTER_MODEL if self.can_fallback else None,
            "since": self._fallback_since or None,
        }

    # -- calls --------------------------------------------------------------

    async def _create(self, kwargs: dict):
        if not self.using_fallback:
            try:
                return await self._anthropic.messages.create(**kwargs)
            except Exception as e:
                if not (_is_out_of_credit(e) and self.can_fallback):
                    raise
                self._switch(e)
        return await self._openrouter_create(kwargs)

    def _openrouter_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._openrouter_key}",
            "Content-Type": "application/json",
        }

    # The Stealth provider behind OPENROUTER_MODEL answers HTTP 200 with an
    # empty message and native_finish_reason "network_error" on roughly two
    # in five calls whenever `tools` is on the request — which is every call
    # JARVIS makes, since plugins are always declared. Measured directly
    # against the endpoint: no such failure appeared once across repeated
    # calls with tools omitted. Retrying is the workaround, not a fix — there
    # is nothing wrong to fix on this side, the fault is upstream of
    # OpenRouter itself. Empty responses are cheap for OpenRouter to redo,
    # unlike the Anthropic 400 above, so a few attempts here are worth it.
    _EMPTY_RESPONSE_RETRIES = 3

    async def _openrouter_create(self, kwargs: dict):
        result = None
        for attempt in range(self._EMPTY_RESPONSE_RETRIES):
            if attempt:
                # Measured bursty, not uniform: back-to-back retries with no
                # gap hit the same congested window and can all come back
                # empty. A short, growing pause gives the shared pool a beat
                # to clear before trying again.
                await asyncio.sleep(0.6 * attempt)
            result = await self._openrouter_create_once(kwargs)
            if result.content:
                return result
            log.warning(
                f"OpenRouter returned no content (attempt {attempt + 1}/"
                f"{self._EMPTY_RESPONSE_RETRIES}) — retrying")
        return result

    async def _openrouter_create_once(self, kwargs: dict):
        body = _to_openrouter(kwargs)
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                OPENROUTER_ENDPOINT, json=body, headers=self._openrouter_headers()
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"OpenRouter {response.status_code}: {response.text[:200]}")
            return _from_openrouter(response.json(), OPENROUTER_MODEL)

    async def _openrouter_text_stream(self, kwargs: dict):
        """Yield text deltas, matching Anthropic's .text_stream contract.

        Retries an empty attempt in full (see _EMPTY_RESPONSE_RETRIES above) —
        safe because nothing has been yielded to the caller yet when an
        attempt turns out empty, so a retry is invisible upstream.
        """
        for attempt in range(self._EMPTY_RESPONSE_RETRIES):
            if attempt:
                await asyncio.sleep(0.6 * attempt)
            got_any = False
            async for chunk in self._openrouter_text_stream_once(kwargs):
                got_any = True
                yield chunk
            if got_any:
                return
            log.warning(
                f"OpenRouter stream returned no text (attempt {attempt + 1}/"
                f"{self._EMPTY_RESPONSE_RETRIES}) — retrying")

    async def _openrouter_text_stream_once(self, kwargs: dict):
        body = _to_openrouter(kwargs)
        body["stream"] = True
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", OPENROUTER_ENDPOINT, json=body, headers=self._openrouter_headers(),
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode(errors="replace")
                    raise RuntimeError(f"OpenRouter {response.status_code}: {detail[:200]}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    try:
                        payload = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    for choice in payload.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]
