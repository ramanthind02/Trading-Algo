"""Integration tests for the multi-source reconciliation layer (Phase 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_platform.core import InstrumentCatalog, load_catalog
from data_platform.core.enums import BarAggregation, InstrumentClass
from data_platform.core.reconciler import SourcePriorityReconciler
from data_platform.core.source_priority import SourcePriorityConfig


def _daily_frame(start: str, n: int, base: float) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n)
    close = base + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100},
        index=pd.DatetimeIndex(idx, name="datetime"),
    )


# ── SourcePriorityConfig ──────────────────────────────────────────────────

def test_default_priority_routes_futures_to_norgate_when_active() -> None:
    cfg = SourcePriorityConfig.default()
    assert cfg.active_source(InstrumentClass.FUTURE, BarAggregation.DAY) == "norgate"


def test_handover_flip_returns_no_source() -> None:
    """IB data pipeline retired — no fallback source configured after Norgate."""
    cfg = SourcePriorityConfig.default()
    cfg.norgate_active = False
    assert cfg.active_source(InstrumentClass.FUTURE, BarAggregation.DAY) is None


def test_cfd_routes_to_mt5_regardless_of_norgate_flag() -> None:
    cfg = SourcePriorityConfig.default()
    assert cfg.active_source(InstrumentClass.CFD, BarAggregation.DAY) == "mt5"
    cfg.norgate_active = False
    assert cfg.active_source(InstrumentClass.CFD, BarAggregation.MONTH) == "mt5"


def test_yaml_config_loads_and_matches_default(tmp_path) -> None:
    # The committed configs/source_priority.yaml should parse and behave like default.
    cfg = SourcePriorityConfig.load()
    assert cfg.active_source(InstrumentClass.FUTURE, BarAggregation.DAY) == "norgate"
    assert cfg.active_source(InstrumentClass.SPOT, BarAggregation.DAY) == "norgate"


# ── reconciler ─────────────────────────────────────────────────────────────

@pytest.fixture
def catalog() -> InstrumentCatalog:
    cat = load_catalog()
    if cat.find("ES.XCME") is None:
        pytest.skip("catalog not seeded (run to_catalog) — integration fixture unavailable")
    return cat


def test_ib_batch_skipped_when_no_ib_rule(catalog: InstrumentCatalog) -> None:
    """IB source retired — passing an 'ib' batch when no rule is configured returns a skip."""
    cfg = SourcePriorityConfig.default()
    cfg.norgate_active = False  # no fallback source configured
    rec = SourcePriorityReconciler(cfg, catalog)

    norgate = _daily_frame("2026-01-01", 10, base=5000.0)
    ib = _daily_frame("2026-01-16", 5, base=4500.0)

    result = rec.reconcile_daily_batch("ES.XCME", norgate, {"ib": ib}, resolution="D")

    assert result.active_source is None
    assert result.skip_reason == "no_active_source_batch"
    assert result.merged is norgate  # existing frame returned unchanged


def test_norgate_active_passthrough_no_ratio(catalog: InstrumentCatalog) -> None:
    """While Norgate is active, futures daily routes to norgate with no ratio scaling."""
    cfg = SourcePriorityConfig.default()  # norgate_active = True
    rec = SourcePriorityReconciler(cfg, catalog)

    existing = _daily_frame("2026-01-01", 10, base=5000.0)
    new = _daily_frame("2026-01-16", 5, base=5010.0)

    result = rec.reconcile_daily_batch("ES.XCME", existing, {"norgate": new}, resolution="D")
    assert result.active_source == "norgate"
    assert result.provenance[0].ratio_applied is False
    # appended rows are unscaled (close stays at ~5010+)
    appended = result.merged.loc[result.merged.index > existing.index[-1]]
    assert float(appended["close"].iloc[0]) == pytest.approx(5010.0, rel=1e-6)


def test_conflict_detected_on_overlap_deviation(catalog: InstrumentCatalog) -> None:
    cfg = SourcePriorityConfig.default()  # norgate_active=True -> routes FUTURE/D to norgate
    rec = SourcePriorityReconciler(cfg, catalog, conflict_threshold=0.005)

    existing = _daily_frame("2026-01-01", 5, base=5000.0)
    # incoming overlaps the same dates but with a >0.5% deviation
    overlap = existing.copy()
    overlap["close"] = overlap["close"] * 1.02  # +2%
    result = rec.reconcile_daily_batch("ES.XCME", existing, {"norgate": overlap}, resolution="D")
    assert len(result.conflicts) > 0
    assert result.conflicts[0].deviation > 0.005


# ── provenance (Phase 3) ───────────────────────────────────────────────────

def test_provenance_write_read_roundtrip(tmp_path, monkeypatch, catalog: InstrumentCatalog) -> None:
    import data_platform.core.provenance as prov
    monkeypatch.setattr(prov, "provenance_dir", lambda: tmp_path)

    cfg = SourcePriorityConfig.default()  # norgate_active=True -> norgate source
    rec = SourcePriorityReconciler(cfg, catalog)
    existing = _daily_frame("2026-01-01", 10, base=5000.0)
    new_bars = _daily_frame("2026-01-16", 5, base=5010.0)
    result = rec.reconcile_daily_batch("ES.XCME", existing, {"norgate": new_bars}, resolution="D")

    prov.write_provenance(result.provenance, resolution="D")
    out = tmp_path / "ES_XCME" / "D_provenance.parquet"
    assert out.is_file()
    df = pd.read_parquet(out)
    assert df.iloc[0]["source"] == "norgate"
    assert bool(df.iloc[0]["ratio_applied"]) is False


def test_log_conflicts_fail_fast_raises(catalog: InstrumentCatalog) -> None:
    import data_platform.core.provenance as prov
    from data_platform.core.reconciler import ConflictRecord
    c = ConflictRecord("ES.XCME", pd.Timestamp("2026-01-02"), "existing", "incoming",
                       5000.0, 5200.0, 0.04)
    with pytest.raises(ValueError):
        prov.log_conflicts((c,), fail_fast=True)
    # warn mode does not raise
    prov.log_conflicts((c,), fail_fast=False)
