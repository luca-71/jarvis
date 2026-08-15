"""Unit tests for battery — pmset parsing and the alert-once-per-drop rules."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import battery

DISCHARGING = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=23461987)\t37%; discharging; 17:30 remaining present: true\n"
)
CHARGING = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=23461987)\t82%; charging; 0:44 remaining present: true\n"
)
CHARGED = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=23461987)\t100%; charged; 0:00 remaining present: true\n"
)
ESTIMATING = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=23461987)\t55%; discharging; (no estimate) present: true\n"
)
DESKTOP = "Now drawing from 'AC Power'\n"


# -- parsing ---------------------------------------------------------------

def test_parses_discharging():
    state = battery._parse(DISCHARGING)
    assert state.percent == 37
    assert state.plugged is False
    assert state.charging is False
    assert state.minutes_remaining == 17 * 60 + 30


def test_parses_charging():
    state = battery._parse(CHARGING)
    assert state.percent == 82
    assert state.plugged is True
    assert state.charging is True


def test_parses_charged_as_plugged_but_not_charging():
    state = battery._parse(CHARGED)
    assert state.percent == 100
    assert state.plugged is True
    assert state.charging is False


def test_zero_estimate_is_not_treated_as_imminent_death():
    """'0:00 remaining' means macOS is still calculating, not 'dies now'."""
    assert battery._parse(CHARGED).minutes_remaining is None


def test_no_estimate_yet():
    state = battery._parse(ESTIMATING)
    assert state.percent == 55
    assert state.minutes_remaining is None


def test_desktop_has_no_battery():
    assert battery._parse(DESKTOP) is None


def test_empty_output():
    assert battery._parse("") is None


# -- alert rules -----------------------------------------------------------

def _monitor():
    return battery.BatteryMonitor(on_low=None)


def _state(percent, plugged=False):
    return battery.BatteryState(
        percent=percent, plugged=plugged, charging=plugged, minutes_remaining=None
    )


def _feed(monitor, *percents, plugged=False):
    """Push readings through exactly as the poll loop does."""
    out = []
    for p in percents:
        state = _state(p, plugged=plugged)
        previous = monitor._last_percent
        monitor._last_percent = state.percent
        out.append(monitor._thresholds_to_speak(state, previous))
    return out


def test_first_reading_only_arms():
    """Starting up already low must not open with a warning."""
    assert _feed(_monitor(), 15) == [None]


def test_announces_on_crossing_twenty():
    assert _feed(_monitor(), 30, 19) == [None, 20]


def test_does_not_repeat_the_same_threshold():
    assert _feed(_monitor(), 30, 19, 18, 17) == [None, 20, None, None]


def test_announces_each_threshold_once_on_the_way_down():
    assert _feed(_monitor(), 30, 19, 9, 4) == [None, 20, 10, 5]


def test_skipped_threshold_is_not_announced_late():
    """A drop from 25 to 8 says '10' once — not '20' now and '10' next poll."""
    assert _feed(_monitor(), 25, 8, 7) == [None, 10, None]


def test_plugging_in_silences_and_rearms():
    monitor = _monitor()
    assert _feed(monitor, 30, 19) == [None, 20]
    # On the charger: nothing said, thresholds re-armed.
    assert _feed(monitor, 25, plugged=True) == [None]
    # Unplugged again and dropping past 20 — worth saying a second time.
    assert _feed(monitor, 19) == [20]


def test_small_recovery_does_not_rearm():
    """A reading bouncing over the line must not speak twice."""
    monitor = _monitor()
    assert _feed(monitor, 30, 19) == [None, 20]
    # 21 is inside REARM_MARGIN of 20, so 20 stays spoken.
    assert _feed(monitor, 21, 19) == [None, None]


def test_meaningful_recovery_rearms():
    monitor = _monitor()
    assert _feed(monitor, 30, 19) == [None, 20]
    assert _feed(monitor, 30, 19) == [None, 20]


def test_never_announces_while_plugged():
    assert _feed(_monitor(), 30, 4, plugged=True) == [None, None]


@pytest.mark.asyncio
async def test_threshold_is_not_burned_with_nobody_listening(monkeypatch):
    """A warning spoken into an empty room is a warning lost for good."""
    announced = []
    listening = False

    readings = iter([_state(30), _state(15), _state(14), _state(13)])

    async def fake_read():
        return next(readings, _state(13))

    async def on_low(state, level):
        announced.append(level)

    monkeypatch.setattr(battery, "read", fake_read)
    monkeypatch.setattr(battery, "POLL_SECONDS", 0.02)

    monitor = battery.BatteryMonitor(on_low, has_listener=lambda: listening)
    monitor.start()
    await asyncio.sleep(0.15)
    assert announced == [], "announced with nobody connected"

    listening = True
    await asyncio.sleep(0.15)
    monitor.stop()

    assert announced == [20], "the held warning never arrived"


# -- prompt context --------------------------------------------------------

def test_prompt_context_empty_before_any_reading(monkeypatch):
    monkeypatch.setattr(battery, "_last_state", None)
    assert battery.format_for_prompt() == ""


def test_prompt_context_reports_percent_and_estimate(monkeypatch):
    monkeypatch.setattr(battery, "_last_state", battery._parse(DISCHARGING))
    monkeypatch.setattr(battery, "_last_read_at", 1e18)
    out = battery.format_for_prompt()
    assert "37%" in out and "on battery" in out and "1050 minutes" in out


def test_prompt_context_omits_estimate_when_charging(monkeypatch):
    monkeypatch.setattr(battery, "_last_state", battery._parse(CHARGING))
    monkeypatch.setattr(battery, "_last_read_at", 1e18)
    out = battery.format_for_prompt()
    assert "charging" in out and "minutes left" not in out


def test_stale_reading_is_dropped(monkeypatch):
    """A reading from an hour ago is worse than no reading."""
    monkeypatch.setattr(battery, "_last_state", battery._parse(DISCHARGING))
    monkeypatch.setattr(battery, "_last_read_at", 0.0)
    assert battery.format_for_prompt() == ""
