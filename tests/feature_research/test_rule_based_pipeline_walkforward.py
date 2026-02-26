from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd

from feature_research.config import FeatureType
from feature_research.in_sample.config import ResearchConfig
from feature_research.pipeline import run_eda_pipeline
from feature_research.walkforward.config import WalkforwardResearchConfig
from feature_research.walkforward.runner import WalkforwardRunReport
from utils.core.enums import Ticker, TimeFrame


def _build_config(tmp_path: Path, *, walkforward_enabled: bool) -> ResearchConfig:
    walkforward = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        enabled=walkforward_enabled,
        test_step=20,
        num_steps=2,
        top_k=2,
        objective_metric_name="mean_return",
        min_fold_samples=10,
        output_root=tmp_path / "shared_results",
    )
    return ResearchConfig(
        feature_type=FeatureType.RULE_BASED,
        tickers=[Ticker.ES],
        start=datetime(2020, 1, 1),
        end=datetime(2020, 4, 29),
        bias_spec={
            "module_name": "rsi_signal",
            "timeframes": [TimeFrame.D],
            "params": {
                "rsi_period": [2, 3],
                "oversold": 25.0,
                "overbought": 65.0,
                "strategy_mode": "long",
                "exit_policy": "threshold_or_bars",
                "exit_bars": 5,
            },
        },
        target_col="log_return",
        strategy="long",
        use_cache=True,
        populate_cache=False,
        reports_dir=tmp_path / "reports",
        walkforward=walkforward,
    )


def _mock_eda_report() -> SimpleNamespace:
    stats_by_level = {
        -1: SimpleNamespace(sharpe=0.1),
        0: SimpleNamespace(sharpe=0.0),
        1: SimpleNamespace(sharpe=0.2),
    }
    return SimpleNamespace(
        rule_stats=SimpleNamespace(per_level_stats=SimpleNamespace(stats_by_level=stats_by_level)),
        diagnostics=SimpleNamespace(is_viable=True, red_flags=[], warnings=[]),
    )


def _series_for_combo(
    params: dict[str, object],
    *,
    tz: str | None = None,
) -> tuple[pd.Series, pd.Series, str]:
    index = pd.date_range("2020-01-01", periods=120, freq="D", tz=tz)
    period = int(cast(int, params["rsi_period"]))
    feature = pd.Series([float(period)] * len(index), index=index, name=f"feature_{period}")
    target = pd.Series([0.01] * len(index), index=index, name="target")
    return feature, target, feature.name


def test_walkforward_enabled_handles_tz_aware_feature_indices(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _build_config(tmp_path, walkforward_enabled=True)

    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 2,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            }
        ],
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.load_features_for_combo",
        lambda single_spec, _config: _series_for_combo(single_spec["params"], tz="UTC"),
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.run_eda_for_rule_based_feature",
        lambda *_args, **_kwargs: _mock_eda_report(),
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.save_eda_report",
        lambda report, output_dir, overwrite: output_dir,
    )

    run_eda_pipeline(config=config, output_dir=tmp_path / "rule_based_reports")

    walkforward_dir = (
        config.walkforward.output_root / "rule_based" / config.bias_spec["module_name"] / "walkforward"
    )
    assert walkforward_dir.exists()


def test_walkforward_disabled_skips_shared_runner(monkeypatch, tmp_path: Path) -> None:
    config = _build_config(tmp_path, walkforward_enabled=False)

    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 2,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            }
        ],
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.load_features_for_combo",
        lambda single_spec, _config: _series_for_combo(single_spec["params"]),
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.run_eda_for_rule_based_feature",
        lambda *_args, **_kwargs: _mock_eda_report(),
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.save_eda_report",
        lambda report, output_dir, overwrite: output_dir,
    )

    calls = {"runner": 0, "stability": 0, "timeline": 0, "writer": 0}
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.run_walkforward_research",
        lambda *_args, **_kwargs: calls.__setitem__("runner", calls["runner"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.plot_selection_stability",
        lambda *_args, **_kwargs: calls.__setitem__("stability", calls["stability"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.plot_fold_timeline",
        lambda *_args, **_kwargs: calls.__setitem__("timeline", calls["timeline"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.write_walkforward_artifacts",
        lambda *_args, **_kwargs: calls.__setitem__("writer", calls["writer"] + 1),
        raising=False,
    )

    run_eda_pipeline(config=config, output_dir=tmp_path / "rule_based_reports")

    assert calls == {"runner": 0, "stability": 0, "timeline": 0, "writer": 0}


def test_walkforward_enabled_writes_selected_feature_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _build_config(tmp_path, walkforward_enabled=True)
    output_dir = tmp_path / "rule_based_reports"

    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 2,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 3,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
        ],
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.load_features_for_combo",
        lambda single_spec, _config: _series_for_combo(single_spec["params"]),
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.run_eda_for_rule_based_feature",
        lambda *_args, **_kwargs: _mock_eda_report(),
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.save_eda_report",
        lambda report, output_dir, overwrite: output_dir,
    )

    run_eda_pipeline(config=config, output_dir=output_dir)

    walkforward_dir = (
        config.walkforward.output_root / "rule_based" / config.bias_spec["module_name"] / "walkforward"
    )
    assert walkforward_dir.exists()
    assert "rule_based/rsi_signal/walkforward" in walkforward_dir.as_posix()

    selection_summary_csv = walkforward_dir / "tables" / "selection_summary.csv"
    assert selection_summary_csv.exists()
    selection_summary_df = pd.read_csv(selection_summary_csv)
    assert "selected_feature" in selection_summary_df.columns


def test_run_rule_based_walkforward_pipeline_returns_report_and_writes_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from feature_research.pipeline import run_rule_based_walkforward_pipeline

    config = _build_config(tmp_path, walkforward_enabled=True)

    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 2,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 3,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
        ],
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.load_features_for_combo",
        lambda single_spec, _config: _series_for_combo(single_spec["params"]),
    )

    report = run_rule_based_walkforward_pipeline(config, tmp_path / "wf_out")

    assert isinstance(report, WalkforwardRunReport)
    walkforward_dir = (
        config.walkforward.output_root / "rule_based" / config.bias_spec["module_name"] / "walkforward"
    )
    assert walkforward_dir.exists()
    assert (walkforward_dir / "tables" / "selection_summary.csv").exists()
    assert (walkforward_dir / "tables" / "fold_scores.csv").exists()
    assert (walkforward_dir / "tables" / "folds.csv").exists()


def test_run_rule_based_walkforward_pipeline_raises_if_no_combos_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from feature_research.pipeline import run_rule_based_walkforward_pipeline
    import pytest

    config = _build_config(tmp_path, walkforward_enabled=True)

    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 2,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            }
        ],
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.load_features_for_combo",
        lambda single_spec, _config: None,
    )

    with pytest.raises(ValueError, match="No param combos loaded successfully"):
        run_rule_based_walkforward_pipeline(config, tmp_path / "wf_out")


def test_run_rule_based_walkforward_pipeline_raises_if_no_folds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from feature_research.pipeline import run_rule_based_walkforward_pipeline
    import pytest

    config = _build_config(tmp_path, walkforward_enabled=True)
    empty_fold_config = dataclasses.replace(
        config,
        walkforward=dataclasses.replace(
            config.walkforward,
            train_start=datetime(2025, 1, 1),
            train_end=datetime(2025, 2, 1),
        ),
    )

    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 2,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            }
        ],
    )
    monkeypatch.setattr(
        "feature_research.in_sample.pipeline.load_features_for_combo",
        lambda single_spec, _config: _series_for_combo(single_spec["params"]),
    )

    with pytest.raises(ValueError, match="No walkforward folds were generated"):
        run_rule_based_walkforward_pipeline(empty_fold_config, tmp_path / "wf_out")
