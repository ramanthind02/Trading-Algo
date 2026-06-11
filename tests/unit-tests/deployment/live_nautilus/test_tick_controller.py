"""Unit tests for the demand-driven tick controller (no Nautilus node)."""
from __future__ import annotations

from deployment.live.runtime.tick_controller import DemandTickController


def _controller() -> tuple[DemandTickController, list, list]:
    subs: list = []
    unsubs: list = []
    ctrl: DemandTickController = DemandTickController(subs.append, unsubs.append)
    return ctrl, subs, unsubs


def test_open_window_subscribes_and_awaits() -> None:
    ctrl, subs, _ = _controller()
    ctrl.open_window(["A", "B"])
    assert sorted(subs) == ["A", "B"]
    assert ctrl.active()
    assert not ctrl.ready()
    assert ctrl.missing() == {"A", "B"}


def test_ready_only_when_all_quoted() -> None:
    ctrl, _, _ = _controller()
    ctrl.open_window(["A", "B"])
    ctrl.record("A", quote=1.0)
    assert not ctrl.ready()
    assert ctrl.missing() == {"B"}
    ctrl.record("B", quote=2.0)
    assert ctrl.ready()
    assert ctrl.latest("A") == 1.0
    assert ctrl.quotes() == {"A": 1.0, "B": 2.0}


def test_open_window_is_idempotent() -> None:
    ctrl, subs, _ = _controller()
    ctrl.open_window(["A"])
    ctrl.record("A", 1.0)
    ctrl.open_window(["A", "B"])  # A already live -> not re-subscribed
    assert subs == ["A", "B"]
    # A was reset to awaiting (prior quote dropped): not ready until re-quoted.
    assert not ctrl.ready()
    assert ctrl.missing() == {"A", "B"}


def test_record_ignored_for_unsubscribed() -> None:
    ctrl, _, _ = _controller()
    ctrl.open_window(["A"])
    ctrl.record("Z", 9.0)  # never subscribed
    assert ctrl.latest("Z") is None


def test_close_window_unsubscribes_all_and_resets() -> None:
    ctrl, _, unsubs = _controller()
    ctrl.open_window(["A", "B"])
    ctrl.record("A", 1.0)
    ctrl.close_window()
    assert sorted(unsubs) == ["A", "B"]
    assert not ctrl.active()
    assert not ctrl.ready()
    assert ctrl.quotes() == {}
