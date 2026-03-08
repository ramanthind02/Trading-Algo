from datetime import datetime
from itertools import product
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest

from feature_research.in_sample.config import ResearchConfig
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    get_effective_range_and_tickers,
    load_features_for_combo,
    param_combo_label,
)
from utils.core.enums import TimeFrame, Ticker


def test_expand_bias_specs_single_param():
    spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": [2, 3]},
    }
    result = expand_bias_specs(spec)
    assert len(result) == 2
    assert result[0]["params"] == {"lookback": 2}
    assert result[1]["params"] == {"lookback": 3}
    assert all(r["module_name"] == "rsi" for r in result)


def test_expand_bias_specs_grid():
    spec = {
        "module_name": "cmma",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": [20, 50], "atr_length": [14, 21]},
    }
    result = expand_bias_specs(spec)
    assert len(result) == 4  # 2x2 grid


def test_expand_bias_specs_scalar_params():
    """Scalar param values (not lists) should be treated as single-element lists."""
    spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": 5},
    }
    result = expand_bias_specs(spec)
    assert len(result) == 1
    assert result[0]["params"] == {"lookback": 5}


def test_param_combo_label_single():
    assert param_combo_label({"lookback": 5}) == "lookback_5"


def test_param_combo_label_multi():
    label = param_combo_label({"lookback": 20, "atr_length": 14})
    # deterministic: sorted keys
    assert label == "atr_length_14__lookback_20"


def test_load_features_for_combo_rejects_raw_return_with_multi_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features_df = pd.DataFrame(
        {
            "rsi_signal_D_lookback_5": [45.0, 55.0],
            "ticker": ["ES", "NQ"],
        }
    )
    targets_df = pd.DataFrame(
        {
            "log_return": [0.001, 0.002],
            "ticker": ["ES", "NQ"],
        }
    )

    def _fake_extract_features_for_bias_node(**_: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        return features_df, targets_df

    monkeypatch.setattr(
        "feature_research.in_sample.data_loader.extract_features_for_bias_node",
        _fake_extract_features_for_bias_node,
    )

    from feature_research.config import FeatureType
    config = cast(
        ResearchConfig,
        SimpleNamespace(
            tickers=["ES", "NQ"],
            start=None,
            end=None,
            target_col="log_return",
            use_cache=False,
            feature_type=FeatureType.CONTINUOUS,
        ),
    )

    with pytest.raises(ValueError, match="Raw return targets must not be mixed"):
        load_features_for_combo(
            single_combo_spec={
                "module_name": "rsi",
                "params": {"lookback": 5},
                "timeframes": [TimeFrame.D],
            },
            config=config,
        )


def _config(start: datetime, end: datetime, tickers: list[Ticker]) -> SimpleNamespace:
    return cast(
        ResearchConfig,
        SimpleNamespace(start=start, end=end, tickers=tickers),
    )


def test_get_effective_range_and_tickers_no_ranges_returns_none() -> None:
    """When no OHLC ranges exist, returns None."""
    config = _config(
        datetime(2000, 1, 1), datetime(2025, 12, 31), [Ticker.ES, Ticker.NQ]
    )
    with patch(
        "feature_research.in_sample.data_loader.get_available_date_ranges_for_tickers",
        return_value={},
    ):
        assert get_effective_range_and_tickers(config) is None


def test_get_effective_range_and_tickers_full_coverage_returns_intersection() -> None:
    """When all tickers cover the full config range, returns that range and all tickers."""
    config = _config(
        datetime(2000, 1, 1), datetime(2025, 12, 31), [Ticker.ES, Ticker.NQ]
    )
    start = pd.Timestamp("2000-01-01")
    end = pd.Timestamp("2025-12-31")
    with patch(
        "feature_research.in_sample.data_loader.get_available_date_ranges_for_tickers",
        return_value={Ticker.ES: (start, end), Ticker.NQ: (start, end)},
    ):
        result = get_effective_range_and_tickers(config)
    assert result is not None
    eff_start, eff_end, tickers = result
    assert eff_start == start and eff_end == end
    assert set(tickers) == {Ticker.ES, Ticker.NQ}


def test_get_effective_range_and_tickers_partial_overlap_narrows_range() -> None:
    """When tickers have partial overlap, returns intersection and only covering tickers."""
    config = _config(
        datetime(2000, 1, 1), datetime(2030, 12, 31), [Ticker.ES, Ticker.NQ]
    )
    # ES 2000-2015, NQ 2010-2030 → intersection 2010-2015; both cover it
    es_start, es_end = pd.Timestamp("2000-01-01"), pd.Timestamp("2015-12-31")
    nq_start, nq_end = pd.Timestamp("2010-01-01"), pd.Timestamp("2030-12-31")
    with patch(
        "feature_research.in_sample.data_loader.get_available_date_ranges_for_tickers",
        return_value={
            Ticker.ES: (es_start, es_end),
            Ticker.NQ: (nq_start, nq_end),
        },
    ):
        result = get_effective_range_and_tickers(config)
    assert result is not None
    eff_start, eff_end, tickers = result
    assert eff_start == pd.Timestamp("2010-01-01")
    assert eff_end == pd.Timestamp("2015-12-31")
    assert set(tickers) == {Ticker.ES, Ticker.NQ}


def test_get_effective_range_and_tickers_disjoint_fallback_to_single_ticker() -> None:
    """When ranges are disjoint, fallback to single ticker with largest overlap."""
    config = _config(
        datetime(2000, 1, 1), datetime(2025, 12, 31), [Ticker.ES, Ticker.NQ]
    )
    # ES 2000-2005, NQ 2020-2025 → no intersection; each has 5y overlap
    es_start, es_end = pd.Timestamp("2000-01-01"), pd.Timestamp("2005-12-31")
    nq_start, nq_end = pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31")
    with patch(
        "feature_research.in_sample.data_loader.get_available_date_ranges_for_tickers",
        return_value={
            Ticker.ES: (es_start, es_end),
            Ticker.NQ: (nq_start, nq_end),
        },
    ):
        result = get_effective_range_and_tickers(config)
    assert result is not None
    eff_start, eff_end, tickers = result
    assert len(tickers) == 1
    assert tickers[0] in (Ticker.ES, Ticker.NQ)
    if tickers[0] == Ticker.ES:
        assert eff_start == es_start and eff_end == es_end
    else:
        assert eff_start == nq_start and eff_end == nq_end
