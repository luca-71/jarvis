"""Unit tests for usage cost tracking.

Regression coverage for two bugs found while investigating why credit
consumption looked high for "just testing": _cost_from_tokens used a flat
$0.80/$4.00-per-million rate that didn't match any deployed model, and
cache_read_input_tokens / cache_creation_input_tokens were logged to the
console but never persisted to the usage log or folded into any cost
estimate — so a cache write (1.25x the base rate) was silently priced as if
it cost nothing.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import server


def test_plain_input_priced_at_sonnet_5_list_rate():
    assert server._cost_from_tokens(1_000_000, 0) == 3.00


def test_output_priced_at_sonnet_5_list_rate():
    assert server._cost_from_tokens(0, 1_000_000) == 15.00


def test_cache_read_priced_at_a_tenth_of_input():
    assert round(server._cost_from_tokens(0, 0, cache_read_t=1_000_000), 4) == 0.30


def test_cache_write_priced_at_1_25x_input():
    assert round(server._cost_from_tokens(0, 0, cache_write_t=1_000_000), 4) == 3.75


def test_cache_write_is_never_free():
    """The bug being fixed: a cache write with zero plain input/output tokens
    used to cost $0.00 in every reported estimate."""
    assert server._cost_from_tokens(0, 0, cache_write_t=500_000) > 0


def test_track_usage_persists_cache_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_USAGE_FILE", tmp_path / "usage.jsonl")
    monkeypatch.setattr(server, "_session_tokens", {
        "input": 0, "output": 0, "api_calls": 0, "tts_calls": 0,
        "cache_read": 0, "cache_write": 0,
    })

    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_read_input_tokens = 3600
    usage.cache_creation_input_tokens = 0
    response = MagicMock()
    response.usage = usage

    server.track_usage(response)

    assert server._session_tokens["cache_read"] == 3600
    assert server._session_tokens["cache_write"] == 0

    logged = (tmp_path / "usage.jsonl").read_text().strip()
    assert '"cache_read_tokens": 3600' in logged
    assert '"cache_write_tokens": 0' in logged


def test_get_usage_for_period_sums_cache_fields(monkeypatch, tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    usage_file.write_text(
        '{"ts": 9999999999, "date": "2026-01-01", "type": "api", '
        '"input_tokens": 100, "output_tokens": 20, '
        '"cache_read_tokens": 3600, "cache_write_tokens": 0}\n'
    )
    monkeypatch.setattr(server, "_USAGE_FILE", usage_file)

    totals = server._get_usage_for_period(None)
    assert totals["cache_read_tokens"] == 3600
    assert totals["cache_write_tokens"] == 0


def test_get_usage_for_period_tolerates_pre_cache_tracking_entries(monkeypatch, tmp_path):
    """Log lines written before this fix have no cache_*_tokens fields at
    all — must default to 0, not raise."""
    usage_file = tmp_path / "usage.jsonl"
    usage_file.write_text(
        '{"ts": 9999999999, "date": "2026-01-01", "type": "api", '
        '"input_tokens": 100, "output_tokens": 20}\n'
    )
    monkeypatch.setattr(server, "_USAGE_FILE", usage_file)

    totals = server._get_usage_for_period(None)
    assert totals["cache_read_tokens"] == 0
    assert totals["cache_write_tokens"] == 0
    assert totals["input_tokens"] == 100
