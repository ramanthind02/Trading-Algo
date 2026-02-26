from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from feature_research.config import FeatureType, OOSWindowConfig
from feature_research.in_sample.config import load_config
from feature_research.pipeline import run_oos_pipeline


def test_rule_based_oos_passes_portfolio_inputs_to_runner(monkeypatch, tmp_path) -> None:
    base_config = load_config()
    oos_window = base_config.oos_window or OOSWindowConfig(
        train_start=datetime(2009, 1, 1),
        train_end=datetime(2023, 12, 30),
        test_start=datetime(2024, 1, 1),
        test_end=datetime(2025, 12, 31),
    )
    config = replace(base_config, feature_type=FeatureType.RULE_BASED, oos_window=oos_window)

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

    monkeypatch.setattr("feature_research.pipeline.populate_cache_if_needed", lambda _cfg: None)
    monkeypatch.setattr(
        "feature_research.pipeline.resolve_walkforward_output_dir",
        lambda **_kwargs: tmp_path / "oos",
    )
    monkeypatch.setattr(
        "feature_research.pipeline.expand_bias_specs",
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
        "feature_research.pipeline.load_features_for_combo",
        lambda _single_spec, _cfg: (feature, target, None),
    )
    monkeypatch.setattr(
        "feature_research.pipeline.build_fold_rows_from_explicit_specs",
        lambda *_args, **_kwargs: fold_rows,
    )
    monkeypatch.setattr(
        "feature_research.pipeline.load_candles_for_config",
        lambda _cfg: portfolio_candles,
    )
    monkeypatch.setattr(
        "feature_research.pipeline.plot_selection_stability",
        lambda *_args, **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        "feature_research.pipeline.plot_fold_timeline",
        lambda *_args, **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        "feature_research.pipeline.write_walkforward_artifacts",
        lambda *_args, **_kwargs: None,
    )

    captured: dict[str, object] = {}

    def _mock_run_walkforward_research(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(selection_summary_df=pd.DataFrame(), folds_df=pd.DataFrame())

    monkeypatch.setattr(
        "feature_research.pipeline.run_walkforward_research",
        _mock_run_walkforward_research,
    )

    run_oos_pipeline(config)

    assert captured["research_config"] is config
    assert captured["portfolio_candles_df"] is portfolio_candles
    feature_data_by_combo = captured["feature_data_by_combo"]
    assert isinstance(feature_data_by_combo, dict)
    assert len(feature_data_by_combo) == 1
    only_value = next(iter(feature_data_by_combo.values()))
    assert list(only_value.columns) == ["feature", "target"]
