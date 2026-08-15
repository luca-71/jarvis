"""Unit tests for timers — duration parsing, single-fire claiming, catch-up."""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import timers


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point every test at its own database, never the real jarvis.db."""
    monkeypatch.setattr(timers, "DB_PATH", tmp_path / "test.db")
    timers.init_timer_db()
    yield


# -- parse_seconds ---------------------------------------------------------

def test_parses_plain_seconds():
    assert timers.parse_seconds("600") == 600


def test_parses_float_seconds():
    assert timers.parse_seconds("90.5") == 90.5


def test_parses_suffixed_units():
    assert timers.parse_seconds("10m") == 600
    assert timers.parse_seconds("90s") == 90
    assert timers.parse_seconds("2h") == 7200


def test_parses_compound_duration():
    assert timers.parse_seconds("1h30m") == 5400


def test_tolerates_whitespace_and_case():
    assert timers.parse_seconds("  10 M  ") == 600


def test_rejects_trailing_junk():
    # A label that lost its ||| delimiter must not become a duration.
    assert timers.parse_seconds("10m of pasta") is None


def test_rejects_non_duration():
    assert timers.parse_seconds("pasta") is None
    assert timers.parse_seconds("") is None
    assert timers.parse_seconds(None) is None


def test_rejects_zero_and_negative():
    assert timers.parse_seconds("0") is None
    assert timers.parse_seconds("-5") is None


def test_rejects_absurd_duration():
    assert timers.parse_seconds(str(timers.MAX_DURATION_SECONDS + 1)) is None


# -- storage ---------------------------------------------------------------

def test_create_and_list():
    timers.create_timer(600, "pasta")
    active = timers.active_timers()
    assert len(active) == 1
    assert active[0].label == "pasta"
    assert 599 < active[0].remaining <= 600


def test_active_sorted_soonest_first():
    timers.create_timer(600, "later")
    timers.create_timer(60, "sooner")
    assert [t.label for t in timers.active_timers()] == ["sooner", "later"]


def test_cancel_removes_from_active():
    t = timers.create_timer(600, "pasta")
    assert timers.cancel_timer(t.id) is True
    assert timers.active_timers() == []


def test_cancel_is_not_idempotent_twice():
    t = timers.create_timer(600, "pasta")
    assert timers.cancel_timer(t.id) is True
    assert timers.cancel_timer(t.id) is False


def test_cancel_matching_by_label():
    timers.create_timer(600, "pasta")
    timers.create_timer(600, "laundry")
    cancelled = timers.cancel_matching("pasta")
    assert [t.label for t in cancelled] == ["pasta"]
    assert [t.label for t in timers.active_timers()] == ["laundry"]


def test_cancel_matching_empty_query_cancels_all():
    timers.create_timer(600, "pasta")
    timers.create_timer(600, "laundry")
    assert len(timers.cancel_matching("")) == 2
    assert timers.active_timers() == []


# -- claiming --------------------------------------------------------------

def test_claim_returns_nothing_before_due():
    timers.create_timer(600, "pasta")
    assert timers._claim_due(time.time()) == []


def test_claim_returns_due_timer():
    timers.create_timer(0.01, "pasta")
    claimed = timers._claim_due(time.time() + 1)
    assert len(claimed) == 1
    assert claimed[0][0].label == "pasta"


def test_claim_fires_exactly_once():
    """The whole point of claiming in SQL: a second pass gets nothing."""
    timers.create_timer(0.01, "pasta")
    now = time.time() + 1
    assert len(timers._claim_due(now)) == 1
    assert len(timers._claim_due(now)) == 0


def test_claim_marks_late_when_overdue():
    timers.create_timer(0.01, "pasta")
    # 30s past due but well inside the missed window.
    claimed = timers._claim_due(time.time() + 30)
    assert claimed[0][1] is True


def test_claim_does_not_mark_late_when_punctual():
    timers.create_timer(0.01, "pasta")
    claimed = timers._claim_due(time.time() + 1)
    assert claimed[0][1] is False


def test_very_overdue_timer_is_missed_not_announced():
    """A timer from yesterday must not ring at breakfast."""
    timers.create_timer(1, "pasta")
    claimed = timers._claim_due(time.time() + timers.MISSED_AFTER_SECONDS + 60)
    assert claimed == []
    # ...but it is out of pending, so it never fires later either.
    assert timers.active_timers() == []


def test_cancelled_timer_never_fires():
    t = timers.create_timer(0.01, "pasta")
    timers.cancel_timer(t.id)
    assert timers._claim_due(time.time() + 10) == []


# -- scheduler -------------------------------------------------------------

@pytest.mark.asyncio
async def test_service_fires_due_timer():
    fired = []

    async def on_fire(timer, late):
        fired.append((timer.label, late))

    service = timers.TimerService(on_fire)
    timers.create_timer(0.05, "pasta")
    service.start()
    await asyncio.sleep(0.4)
    service.stop()

    assert fired == [("pasta", False)]


@pytest.mark.asyncio
async def test_service_picks_up_timer_set_after_start():
    """A new timer must wake the loop rather than wait out its long sleep."""
    fired = []

    async def on_fire(timer, late):
        fired.append(timer.label)

    service = timers.TimerService(on_fire)
    service.start()
    await asyncio.sleep(0.05)

    timers.create_timer(0.05, "late arrival")
    service.notify()
    await asyncio.sleep(0.4)
    service.stop()

    assert fired == ["late arrival"]


@pytest.mark.asyncio
async def test_service_fires_timer_left_over_from_a_previous_run():
    """Catch-up: the row outlives the process that created it."""
    fired = []

    async def on_fire(timer, late):
        fired.append((timer.label, late))

    # Written as if by an earlier server that then died.
    timers.create_timer(0.01, "leftover")
    await asyncio.sleep(0.05)

    service = timers.TimerService(on_fire)
    service.start()
    await asyncio.sleep(0.2)
    service.stop()

    assert len(fired) == 1
    assert fired[0][0] == "leftover"


@pytest.mark.asyncio
async def test_due_timer_is_held_until_someone_is_listening(monkeypatch):
    """Claiming is irreversible, so it must not happen into an empty room.

    This is the startup case: the scheduler runs a few hundred milliseconds
    before the first client connects, which is long enough to lose a catch-up
    announcement entirely.
    """
    fired = []
    listening = False

    async def on_fire(timer, late):
        fired.append((timer.label, late))

    monkeypatch.setattr(timers, "LISTENER_POLL_SECONDS", 0.02)
    service = timers.TimerService(on_fire, has_listener=lambda: listening)
    timers.create_timer(0.01, "leftover")
    service.start()

    await asyncio.sleep(0.3)
    assert fired == [], "fired with nobody connected"
    assert len(timers.active_timers()) == 1, "claimed with nobody connected"

    listening = True
    await asyncio.sleep(0.3)
    service.stop()

    assert len(fired) == 1
    assert fired[0][0] == "leftover"


@pytest.mark.asyncio
async def test_held_timer_reports_itself_as_late(monkeypatch):
    """Held past its deadline, it should admit to it rather than pretend."""
    fired = []
    listening = False

    async def on_fire(timer, late):
        fired.append((timer.label, late))

    monkeypatch.setattr(timers, "LISTENER_POLL_SECONDS", 0.02)
    service = timers.TimerService(on_fire, has_listener=lambda: listening)
    # A negative offset backdates fire_at, which is what a genuine leftover
    # looks like: due five minutes ago, written by a process that is gone.
    timers.create_timer(-300, "leftover")
    service.start()
    await asyncio.sleep(0.2)

    listening = True
    await asyncio.sleep(0.3)
    service.stop()

    assert fired == [("leftover", True)]


@pytest.mark.asyncio
async def test_missing_listener_hook_means_always_fire():
    """Callers that do not pass the hook keep the simple behaviour."""
    fired = []

    async def on_fire(timer, late):
        fired.append(timer.label)

    service = timers.TimerService(on_fire)  # no has_listener
    timers.create_timer(0.01, "pasta")
    service.start()
    await asyncio.sleep(0.3)
    service.stop()

    assert fired == ["pasta"]


@pytest.mark.asyncio
async def test_one_failing_announcement_does_not_block_the_next():
    fired = []

    async def on_fire(timer, late):
        if timer.label == "bad":
            raise RuntimeError("TTS exploded")
        fired.append(timer.label)

    timers.create_timer(0.01, "bad")
    timers.create_timer(0.02, "good")

    service = timers.TimerService(on_fire)
    service.start()
    await asyncio.sleep(0.4)
    service.stop()

    assert fired == ["good"]


# -- prompt context --------------------------------------------------------

def test_prompt_context_when_empty():
    assert "No timers" in timers.format_for_prompt()


def test_prompt_context_lists_remaining_seconds():
    timers.create_timer(600, "pasta")
    out = timers.format_for_prompt()
    assert "pasta" in out
    assert "599s remaining" in out or "600s remaining" in out
