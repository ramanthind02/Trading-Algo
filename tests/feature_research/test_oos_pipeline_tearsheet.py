from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from feature_research.config import FeatureType, OOSWindowConfig
from feature_research.in_sample.config import load_config
from feature_research.pipeline import run_oos_pipeline
from utils.core.enums import Ticker


def test_signed_signal_oos_passes_portfolio_inputs_to_runner(monkeypatch, tmp_path) -> None:
    base_config = load_config()
    oos_window = base_config.oos_window or OOSWindowConfig(
        train_start=datetime(2009, 1, 1),
        train_end=datetime(2023, 12, 30),
        test_start=datetime(2024, 1, 1),
        test_end=datetime(2025, 12, 31),
    )
    config = replace(base_config, feature_type=FeatureType.SIGNED_SIGNAL, oos_window=oos_window)

    base_index = pd.date_range("2023-01-01", periods=15, freq="D")
    index = pd.DatetimeIndex(np.repeat(base_index.values, 2))
    feature = pd.Series(1.0, index=index, name="feature")
    target = pd.Series(0.01, index=index, name="target")
    portfolio_candles = pd.DataFrame({"close": 100.0}, index=index)
    fold_rows = [
        {
            "fold_id": 0,
            "train_start": index[0],
            "train_end": index[9],
            "test_start": index[10],
            "test_end": index[-1],
        }
    ]

    monkeypatch.setattr(
        "feature_research.pipelines._shared.populate_cache_if_needed",
        lambda _cfg, **_kwargs: None,
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.get_tickers_with_coverage_for_config",
        lambda _cfg, **_kwargs: _cfg.tickers,
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.resolve_walkforward_output_dir",
        lambda **_kwargs: tmp_path / "oos",
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.expand_bias_specs",
        lambda _spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": ["D"],
                "params": {
                    "rsi_period": 2,
                    "oversold": 25,
                    "overbought": 75,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            }
        ],
    )
    monkeypatch.setattr(
        "utils.evaluation.walkforward.research_data.load_features_for_combo",
        lambda _single_spec, _cfg, **_kwargs: (feature, target, None),
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.build_fold_rows_from_explicit_specs",
        lambda *_args, **_kwargs: fold_rows,
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.load_portfolio_candles",
        lambda _cfg: portfolio_candles,
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.write_walkforward_artifacts",
        lambda *_args, **_kwargs: None,
    )

    captured: dict[str, object] = {}

    def _mock_run_walkforward_research(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(selection_summary_df=pd.DataFrame(), folds_df=pd.DataFrame())

    monkeypatch.setattr(
        "feature_research.pipelines._shared.run_walkforward_research",
        _mock_run_walkforward_research,
    )

    run_oos_pipeline(config)

    assert captured["research_config"] is config
    assert captured["portfolio_candles_df"] is portfolio_candles
    target_forwarded = captured["target"]
    assert isinstance(target_forwarded, pd.Series)
    assert int((target_forwarded != 0).sum()) > 0
    feature_data_by_combo = captured["feature_data_by_combo"]
    assert isinstance(feature_data_by_combo, dict)
    assert len(feature_data_by_combo) == 1
    only_value = next(iter(feature_data_by_combo.values()))
    assert list(only_value.columns) == ["signal", "target"]


def test_oos_pipeline_uses_available_data_when_no_full_coverage(
    monkeypatch, tmp_path
) -> None:
    """When no ticker has full coverage, pipeline narrows to available data and completes."""
    base_config = load_config()
    oos_window = base_config.oos_window or OOSWindowConfig(
        train_start=datetime(2009, 1, 1),
        train_end=datetime(2023, 12, 30),
        test_start=datetime(2024, 1, 1),
        test_end=datetime(2025, 12, 31),
    )
    config = replace(
        base_config, feature_type=FeatureType.SIGNED_SIGNAL, oos_window=oos_window
    )

    base_index = pd.date_range("2023-01-01", periods=15, freq="D")
    index = pd.DatetimeIndex(np.repeat(base_index.values, 2))
    feature = pd.Series(1.0, index=index, name="feature")
    target = pd.Series(0.01, index=index, name="target")
    portfolio_candles = pd.DataFrame({"close": 100.0}, index=index)
    narrowed_start = pd.Timestamp("2020-01-01")
    narrowed_end = pd.Timestamp("2024-06-30")
    fold_spec_captured: list[list[tuple]] = []

    def _capture_fold_rows(
        reference_index: pd.DatetimeIndex,
        explicit_specs: list[tuple],
        *args: object,
        **kwargs: object,
    ) -> list[dict]:
        fold_spec_captured.append(explicit_specs)
        (train_start, train_end, test_start, test_end) = explicit_specs[0]
        return [
            {
                "fold_id": 0,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        ]

    monkeypatch.setattr(
        "feature_research.pipelines._shared.populate_cache_if_needed",
        lambda _cfg, **_kwargs: None,
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.get_tickers_with_coverage_for_config",
        lambda _cfg, **_kwargs: [],
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.get_effective_range_and_tickers",
        lambda _cfg, **_kwargs: (narrowed_start, narrowed_end, [Ticker.TLT]),
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.resolve_walkforward_output_dir",
        lambda **_kwargs: tmp_path / "oos",
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.expand_bias_specs",
        lambda _spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": ["D"],
                "params": {
                    "rsi_period": 2,
                    "oversold": 25,
                    "overbought": 75,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            }
        ],
    )
    monkeypatch.setattr(
        "utils.evaluation.walkforward.research_data.load_features_for_combo",
        lambda _single_spec, _cfg, **_kwargs: (feature, target, None),
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.build_fold_rows_from_explicit_specs",
        _capture_fold_rows,
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.load_portfolio_candles",
        lambda _cfg: portfolio_candles,
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.write_walkforward_artifacts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.run_walkforward_research",
        lambda *_args, **_kwargs: SimpleNamespace(
            selection_summary_df=pd.DataFrame(), folds_df=pd.DataFrame()
        ),
    )

    run_oos_pipeline(config)

    assert len(fold_spec_captured) == 1
    specs = fold_spec_captured[0]
    assert len(specs) == 1
    train_start, train_end, test_start, test_end = specs[0]
    assert train_start >= narrowed_start
    assert test_end <= narrowed_end
