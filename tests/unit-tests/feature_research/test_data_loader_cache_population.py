from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from feature_research.config import CachePopulationMode
from feature_research.in_sample.data_loader import (
    estimate_bias_lookback_buffer_days,
    BIAS_MODULE_COMBO_KEY,
    CachePopulationWindow,
    bias_spec_for_node_instantiation,
    populate_cache_if_needed,
    resolve_cache_population_window,
)
from utils.core.enums import Ticker, TimeFrame


def _config(*, start: datetime, end: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        tickers=[Ticker.NQ],
        bias_spec={
            "module_name": "rsisignal",
            "timeframes": [TimeFrame.D],
            "params": {"lookback": 14},
        },
        start=start,
        end=end,
    )


def test_resolve_cache_population_window_uses_full_common_ohlc() -> None:
    cfg = _config(
        start=datetime(2000, 1, 1),
        end=datetime(2022, 12, 31),
    )
    ranges = {
        Ticker.NQ: (datetime(1990, 6, 1), datetime(2025, 12, 31)),
    }

    window = resolve_cache_population_window(
        cfg,
        ranges=ranges,
        bootstrap_tickers=[Ticker.NQ],
    )

    assert window == CachePopulationWindow(
        bootstrap_start=datetime(1990, 6, 1),
        bootstrap_end=datetime(2025, 12, 31),
        bias_coverage_start=None,
        bias_coverage_end=datetime(2025, 12, 31),
        used_config_fallback=False,
    )


def test_resolve_cache_population_window_analysis_plus_lookback_clips_bias_start() -> None:
    cfg = _config(
        start=datetime(2000, 1, 1),
        end=datetime(2022, 12, 31),
    )
    ranges = {
        Ticker.NQ: (datetime(1990, 6, 1), datetime(2025, 12, 31)),
    }
    buffer_days = estimate_bias_lookback_buffer_days(cfg.bias_spec)

    window = resolve_cache_population_window(
        cfg,
        ranges=ranges,
        bootstrap_tickers=[Ticker.NQ],
        cache_population_mode=CachePopulationMode.ANALYSIS_PLUS_LOOKBACK,
        lookback_buffer_days=buffer_days,
    )

    expected_bias_start = (pd.Timestamp(cfg.start) - pd.Timedelta(days=buffer_days)).to_pydatetime()
    assert window.bias_coverage_start == max(datetime(1990, 6, 1), expected_bias_start)
    assert window.bias_coverage_end == datetime(2025, 12, 31)


def test_resolve_cache_population_window_falls_back_to_config_when_ranges_incomplete() -> None:
    cfg = _config(
        start=datetime(2000, 1, 1),
        end=datetime(2022, 12, 31),
    )

    window = resolve_cache_population_window(
        cfg,
        ranges={},
        bootstrap_tickers=[Ticker.NQ],
    )

    assert window.used_config_fallback is True
    assert window.bootstrap_start == datetime(2000, 1, 1)
    assert window.bootstrap_end == datetime(2022, 12, 31)
    assert window.bias_coverage_start == datetime(2000, 1, 1)
    assert window.bias_coverage_end == datetime(2022, 12, 31)


def test_resolve_cache_population_window_raises_when_analysis_outside_ohlc() -> None:
    cfg = _config(
        start=datetime(1990, 1, 1),
        end=datetime(1995, 12, 31),
    )
    ranges = {
        Ticker.NQ: (datetime(2000, 1, 1), datetime(2022, 12, 31)),
    }

    with pytest.raises(ValueError, match="does not overlap"):
        resolve_cache_population_window(
            cfg,
            ranges=ranges,
            bootstrap_tickers=[Ticker.NQ],
        )


def test_populate_cache_if_needed_passes_full_history_to_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = _config(
        start=datetime(2000, 1, 1),
        end=datetime(2022, 12, 31),
    )
    candle_dir = tmp_path / "data" / "ohlc_data"
    candle_dir.mkdir(parents=True)
    (candle_dir / "NQ_D.parquet").touch()

    captured: dict[str, object] = {}

    class _FakeManager:
        def get_available_date_range_per_ticker(self, *, tickers, timeframes):
            return {
                Ticker.NQ: (datetime(1990, 1, 1), datetime(2025, 12, 31)),
            }

        def bootstrap_source_candles(self, **kwargs):
            captured["bootstrap"] = kwargs
            return {"failed": 0}

        def ensure_bias_cache_coverage(self, **kwargs):
            captured["coverage"] = kwargs
            return {"failed": 0, "total_tasks": 1, "rebuilt": 0, "validated": 1}

    monkeypatch.setattr(
        "feature_research.in_sample.data_loader._resolve_project_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.data_loader.CacheManager",
        lambda candle_dir: _FakeManager(),
    )
    monkeypatch.setattr(
        "feature_research.in_sample.data_loader.expand_bias_specs",
        lambda _spec: [
            {
                "module_name": "rsisignal",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": 14},
            }
        ],
    )

    populate_cache_if_needed(cfg)

    bootstrap = captured["bootstrap"]
    assert bootstrap["start_date"] == datetime(1990, 1, 1)
    assert bootstrap["end_date"] == datetime(2025, 12, 31)

    coverage = captured["coverage"]
    assert coverage["start_date"] is None
    assert coverage["end_date"] == datetime(2025, 12, 31)


def test_populate_cache_if_needed_fallback_uses_config_bounds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = _config(
        start=datetime(2000, 1, 1),
        end=datetime(2022, 12, 31),
    )
    candle_dir = tmp_path / "data" / "ohlc_data"
    candle_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    class _FakeManager:
        def get_available_date_range_per_ticker(self, *, tickers, timeframes):
            return {}

        def bootstrap_source_candles(self, **kwargs):
            captured["bootstrap"] = kwargs
            return {"failed": 0}

        def ensure_bias_cache_coverage(self, **kwargs):
            captured["coverage"] = kwargs
            return {"failed": 0, "total_tasks": 0, "rebuilt": 0, "validated": 0}

    monkeypatch.setattr(
        "feature_research.in_sample.data_loader._resolve_project_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.data_loader.CacheManager",
        lambda candle_dir: _FakeManager(),
    )
    monkeypatch.setattr(
        "feature_research.in_sample.data_loader.expand_bias_specs",
        lambda _spec: [],
    )

    populate_cache_if_needed(cfg)

    assert captured["bootstrap"]["start_date"] == datetime(2000, 1, 1)
    assert captured["bootstrap"]["end_date"] == datetime(2022, 12, 31)
    assert captured["coverage"]["start_date"] == datetime(2000, 1, 1)
    assert captured["coverage"]["end_date"] == datetime(2022, 12, 31)


def test_bias_spec_for_node_instantiation_strips_bias_module_key() -> None:
    spec = {
        "module_name": "cyclical_rsi_signal",
        "timeframes": ["D"],
        "params": {
            "short_period": 4,
            "long_period": 120,
            BIAS_MODULE_COMBO_KEY: "cyclical_rsi_signal",
        },
    }
    cleaned = bias_spec_for_node_instantiation(spec)
    assert BIAS_MODULE_COMBO_KEY not in cleaned["params"]
    assert cleaned["params"]["short_period"] == 4
