from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from research.feature.config import FeatureType, ResearchWindowConfig, VectorShuffleScope, load_config
from lib.core.enums import Ticker, TimeFrame
from research.feature.pipelines.permutation import (
    _require_permutation_enabled,
    run_permutation_pipeline,
)


def test_run_permutation_pipeline_uses_preloaded_signed_signal_data(
    monkeypatch,
) -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    combo_params = [{"lookback": 2}, {"lookback": 3}]
    target = pd.Series([0.01, -0.02, 0.03, 0.01, 0.02], index=index, name="target")
    signal_frames = {
        tuple(sorted(params.items())): pd.DataFrame(
            {
                "signal": pd.Series([float(params["lookback"])] * len(index), index=index, name=f"sig_{params['lookback']}"),
                "target": target,
                "returns": pd.Series([float(params["lookback"])] * len(index), index=index).mul(target).rename("returns"),
            }
        )
        for params in combo_params
    }

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "research.feature.pipelines.permutation.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "research.feature.pipelines.permutation.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsisignal", "timeframes": [], "params": params}
            for params in combo_params
        ],
    )
    monkeypatch.setattr(
        "research.feature.pipelines.permutation.load_signed_signal_research_data",
        lambda _config, _expanded, print_loaded=False: SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=combo_params,
            reference_index=index,
            reference_target_series=target,
        ),
    )
    monkeypatch.setattr(
        "research.feature.pipelines.permutation.load_candles_for_config",
        lambda _config: pd.DataFrame({"close": target}, index=index),
    )

    def _fake_run_suite(**kwargs: object) -> object:
        captured.update(kwargs)
        extractor = kwargs["extractor_func"]
        extracted = extractor(pd.DataFrame({"close": target}, index=index), {"lookback": 2})
        pd.testing.assert_series_equal(extracted, signal_frames[(("lookback", 2),)]["signal"])
        return SimpleNamespace(feature_name="sig_2", feature_type="signed_signal", funnel_stats=SimpleNamespace(total_params=2, stage1_pass=2))

    monkeypatch.setattr(
        "research.feature.pipelines.permutation.run_permutation_test_suite",
        _fake_run_suite,
    )

    base = load_config()
    rsisignal_spec = {
        "module_name": "rsisignal",
        "params": {"lookback": [2, 3]},
        "timeframes": [TimeFrame.D],
    }
    signed_defaults = replace(base.in_sample_defaults.signed_signal, bias_spec=rsisignal_spec)
    catalog = replace(
        base.in_sample_defaults,
        signed_signal=signed_defaults,
        continuous=replace(base.in_sample_defaults.continuous, bias_spec=rsisignal_spec),
    )
    config = replace(
        base,
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 10),
        tickers=[Ticker.ES],
        permutation=replace(base.permutation, vector_shuffle_scope=VectorShuffleScope.FULL_GRID),
        research_window=ResearchWindowConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 1, 5),
            val_start=datetime(2020, 1, 6),
            val_end=datetime(2020, 1, 10),
        ),
        in_sample_defaults=catalog,
    )

    output_dir = Path(".tmp_pytest_run") / "permutation_pipeline_unit"
    output_dir.mkdir(parents=True, exist_ok=True)
    suite, grid = run_permutation_pipeline(config, output_dir)

    assert grid == combo_params
    assert suite is not None
    assert captured["param_grid"] == combo_params
    # Feature shuffle (scalar Stage 1) must not pass pre-aligned batches; those select
    # target-permutation null instead of permuting the feature vector.
    assert captured.get("aligned_signals_by_combo") is None


def test_require_permutation_enabled_raises_when_disabled() -> None:
    cfg = load_config()
    off = replace(cfg, permutation=replace(cfg.permutation, enabled=False))
    with pytest.raises(ValueError, match="disabled"):
        _require_permutation_enabled(off)


def test_run_permutation_pipeline_selected_combo_filters_grid(monkeypatch) -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    combo_params = [{"lookback": 2}, {"lookback": 3}]
    target = pd.Series([0.01, -0.02, 0.03, 0.01, 0.02], index=index, name="target")
    signal_frames = {
        tuple(sorted(params.items())): pd.DataFrame(
            {
                "signal": pd.Series([float(params["lookback"])] * len(index), index=index, name=f"sig_{params['lookback']}"),
                "target": target,
                "returns": pd.Series([float(params["lookback"])] * len(index), index=index).mul(target).rename("returns"),
            }
        )
        for params in combo_params
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "research.feature.pipelines.permutation.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "research.feature.pipelines.permutation.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsisignal", "timeframes": [], "params": params}
            for params in combo_params
        ],
    )
    monkeypatch.setattr(
        "research.feature.pipelines.permutation.load_signed_signal_research_data",
        lambda _config, _expanded, print_loaded=False: SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=combo_params,
            reference_index=index,
            reference_target_series=target,
        ),
    )
    monkeypatch.setattr(
        "research.feature.pipelines.permutation.load_candles_for_config",
        lambda _config: pd.DataFrame({"close": target}, index=index),
    )
    monkeypatch.setattr(
        "research.feature.pipelines.permutation.run_permutation_test_suite",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            feature_name="sig_3",
            feature_type="signed_signal",
            funnel_stats=SimpleNamespace(total_params=1, stage1_pass=1),
        ),
    )

    base = load_config()
    rsisignal_spec = {
        "module_name": "rsisignal",
        "params": {"lookback": [2, 3]},
        "timeframes": [TimeFrame.D],
    }
    signed_defaults = replace(base.in_sample_defaults.signed_signal, bias_spec=rsisignal_spec)
    catalog = replace(
        base.in_sample_defaults,
        signed_signal=signed_defaults,
        continuous=replace(base.in_sample_defaults.continuous, bias_spec=rsisignal_spec),
    )
    config = replace(
        base,
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 10),
        tickers=[Ticker.ES],
        research_window=ResearchWindowConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 1, 5),
            val_start=datetime(2020, 1, 6),
            val_end=datetime(2020, 1, 10),
        ),
        in_sample_defaults=catalog,
        permutation=replace(
            base.permutation,
            vector_shuffle_scope=VectorShuffleScope.SELECTED_COMBO,
        ),
    )

    output_dir = Path(".tmp_pytest_run") / "permutation_pipeline_selected"
    output_dir.mkdir(parents=True, exist_ok=True)
    _, grid = run_permutation_pipeline(
        config,
        output_dir,
        selected_combo_name="lookback_3",
    )

    assert grid == [{"lookback": 3}]
    assert captured["param_grid"] == [{"lookback": 3}]
