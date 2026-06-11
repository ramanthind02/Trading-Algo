"""Unit tests for the durable daily-decision state (restart idempotency)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from deployment.live.runtime import decision_state as ds


def test_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "decision_state.json"
    state = ds.DecisionState(session_date=date(2026, 6, 5), executed=frozenset({"ES", "NQ"}))
    ds.save(p, state)
    loaded = ds.load(p)
    assert loaded.session_date == date(2026, 6, 5)
    assert loaded.executed == frozenset({"ES", "NQ"})


def test_executed_for_same_vs_other_session(tmp_path: Path) -> None:
    state = ds.DecisionState(session_date=date(2026, 6, 5), executed=frozenset({"ES"}))
    # Restart on the SAME session resumes (excludes already-executed).
    assert state.executed_for(date(2026, 6, 5)) == frozenset({"ES"})
    # A new session resets (nothing excluded) -> a fresh day re-decides everything.
    assert state.executed_for(date(2026, 6, 6)) == frozenset()
    assert state.executed_for(None) == frozenset()


def test_missing_file_is_empty(tmp_path: Path) -> None:
    loaded = ds.load(tmp_path / "nope.json")
    assert loaded.session_date is None
    assert loaded.executed == frozenset()


def test_corrupt_file_self_heals(tmp_path: Path) -> None:
    p = tmp_path / "decision_state.json"
    p.write_text("{ this is not valid json", encoding="utf-8")
    loaded = ds.load(p)
    # Empty state (so the day simply re-decides — safe), and the bad file quarantined.
    assert loaded.session_date is None
    assert (tmp_path / "decision_state.json.corrupt").exists()


def test_state_path_per_broker() -> None:
    p_ftmo = ds.state_path_for_broker("ftmo")
    p_dwx = ds.state_path_for_broker("darwinex")
    assert p_ftmo != p_dwx
    assert "ftmo" in p_ftmo.parts and p_ftmo.name == "decision_state.json"


def test_simulated_restart_excludes_executed(tmp_path: Path) -> None:
    """The mechanism that prevents a re-fired decision from double-trading."""
    p = tmp_path / "decision_state.json"
    session = date(2026, 6, 5)
    # Before "crash": ES+NQ executed this session.
    ds.save(p, ds.DecisionState(session_date=session, executed=frozenset({"ES", "NQ"})))
    # After "restart": reload and rebuild the pending set for the SAME session.
    loaded = ds.load(p)
    desired = {"ES": 0.5, "NQ": -0.3, "GC": 0.2}
    already = loaded.executed_for(session)
    pending = {c: f for c, f in desired.items() if c not in already}
    assert set(pending) == {"GC"}  # only the unfinished symbol is retried
