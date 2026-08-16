"""Unit tests for server's sentence-chunked streaming TTS.

Regression/behavior coverage for _pop_complete_sentences (the buffer
splitter) and _stream_and_speak (streams a Claude response and speaks each
sentence as it completes, instead of waiting for the whole reply before any
audio starts).
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import server


# -- _pop_complete_sentences --------------------------------------------------

def test_pops_a_single_complete_sentence():
    sentences, tail = server._pop_complete_sentences("Sir, I found three files. ")
    assert sentences == ["Sir, I found three files."]
    assert tail == ""


def test_leaves_an_incomplete_sentence_in_the_tail():
    sentences, tail = server._pop_complete_sentences("Sir, I found three")
    assert sentences == []
    assert tail == "Sir, I found three"


def test_pops_multiple_sentences_arriving_in_one_delta():
    sentences, tail = server._pop_complete_sentences(
        "Sir, I found three files. Shall I proceed? "
    )
    assert sentences == ["Sir, I found three files.", "Shall I proceed?"]
    assert tail == ""


def test_boundary_split_across_two_deltas_is_not_lost():
    # First delta ends with punctuation but no trailing whitespace yet —
    # must not fire until the next delta confirms the boundary.
    sentences, tail = server._pop_complete_sentences("Sir, I found three files.")
    assert sentences == []
    assert tail == "Sir, I found three files."

    sentences, tail = server._pop_complete_sentences(tail + " Shall I proceed?")
    assert sentences == ["Sir, I found three files."]
    assert tail == "Shall I proceed?"


def test_empty_buffer_yields_nothing():
    sentences, tail = server._pop_complete_sentences("")
    assert sentences == []
    assert tail == ""


# -- _stream_and_speak ---------------------------------------------------

class _FakeStream:
    """Mimics the async context manager returned by messages.stream()."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def _gen():
            for chunk in self._chunks:
                yield chunk
        return _gen()


@pytest.fixture
def fake_ws():
    ws = MagicMock()
    ws.send_json = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_speaks_each_sentence_as_it_completes(monkeypatch, fake_ws):
    chunks = ["Sir, I found three files. ", "Shall I proceed?"]
    fake_client = MagicMock()
    fake_client.messages.stream = MagicMock(return_value=_FakeStream(chunks))
    monkeypatch.setattr(server, "anthropic_client", fake_client)
    monkeypatch.setattr(server, "synthesize_speech", AsyncMock(return_value=b"audio-bytes"))

    result = await server._stream_and_speak(fake_ws, "claude-haiku-4-5-20251001", 700, "sys", [])

    assert result == "Sir, I found three files. Shall I proceed?"
    # First call puts the UI into "speaking" before anything is synthesized.
    assert fake_ws.send_json.call_args_list[0].args[0] == {"type": "status", "state": "speaking"}
    audio_calls = [c for c in fake_ws.send_json.call_args_list if c.args[0].get("type") == "audio"]
    assert len(audio_calls) == 2
    assert audio_calls[0].args[0]["text"] == "Sir, I found three files. "
    assert audio_calls[1].args[0]["text"] == "Sir, I found three files. Shall I proceed?"


@pytest.mark.asyncio
async def test_trailing_text_without_terminal_punctuation_is_still_spoken(monkeypatch, fake_ws):
    chunks = ["Sir, all done"]  # no sentence-ending punctuation at all
    fake_client = MagicMock()
    fake_client.messages.stream = MagicMock(return_value=_FakeStream(chunks))
    monkeypatch.setattr(server, "anthropic_client", fake_client)
    speak_mock = AsyncMock(return_value=b"audio-bytes")
    monkeypatch.setattr(server, "synthesize_speech", speak_mock)

    result = await server._stream_and_speak(fake_ws, "claude-haiku-4-5-20251001", 700, "sys", [])

    assert result == "Sir, all done"
    speak_mock.assert_awaited_once()
    audio_calls = [c for c in fake_ws.send_json.call_args_list if c.args[0].get("type") == "audio"]
    assert len(audio_calls) == 1


@pytest.mark.asyncio
async def test_failed_synthesis_falls_back_to_text_message(monkeypatch, fake_ws):
    chunks = ["Sir, all done."]
    fake_client = MagicMock()
    fake_client.messages.stream = MagicMock(return_value=_FakeStream(chunks))
    monkeypatch.setattr(server, "anthropic_client", fake_client)
    monkeypatch.setattr(server, "synthesize_speech", AsyncMock(return_value=None))

    await server._stream_and_speak(fake_ws, "claude-haiku-4-5-20251001", 700, "sys", [])

    text_calls = [c for c in fake_ws.send_json.call_args_list if c.args[0].get("type") == "text"]
    assert len(text_calls) == 1
    assert text_calls[0].args[0]["text"] == "Sir, all done."
