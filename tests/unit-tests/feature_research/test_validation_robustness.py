from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from feature_research.config import EvaluationDefaultsCatalog, EvaluationPhaseDefaultsConfig, ResearchWindowConfig, load_config
from feature_research._internal.core_helpers import combo_key
from feature_research.filter_research_labels import research_display_label
from feature_research.in_sample.data_loader import enrich_param_combo_with_module, expand_bias_specs
from feature_research.validation.robustness_runner import (
    _params_match,
    _rank_scatter_chosen_params,
    _resolve_chosen_params,
    load_validation_robustness_report_from_json,
    run_and_write_validation_robustness,
    run_validation_robustness_pipeline,
    write_validation_robustness_summary,
)


def _build_combo_frame(
    index: pd.DatetimeIndex,
    target: pd.Series,
    signal_level: float,
    *,
    signal_name: str,
    alternating: bool = False,
) -> pd.DataFrame:
    if alternating:
        signal_values = [
            signal_level if position % 2 == 0 else -signal_level * 0.5
            for position in range(len(index))
        ]
    else:
        signal_values = [signal_level for _ in range(len(index))]
    signal = pd.Series(signal_values, index=index, name=signal_name, dtype=float)
    returns = signal.mul(target).rename("returns")
    return pd.DataFrame(
        {
            "signal": signal,
            "target": target,
            "returns": returns,
        }
    )


def _validation_test_config(
    *,
    combo_params: list[dict[str, int]],
    eval_params: dict[str, int],
    index: pd.DatetimeIndex,
) -> object:
    base = load_config()
    eval_bias_spec = {
        "module_name": "rsisignal",
        "timeframes": ["D"],
        "params": eval_params,
    }
    return replace(
        base,
        start=index[0].to_pydatetime(),
        end=index[-1].to_pydatetime(),
        research_window=ResearchWindowConfig(
            train_start=index[0].to_pydatetime(),
            train_end=index[120].to_pydatetime(),
            val_start=index[121].to_pydatetime(),
            val_end=index[-1].to_pydatetime(),
        ),
        in_sample_defaults=replace(
            base.in_sample_defaults,
            signed_signal=replace(
                base.in_sample_defaults.signed_signal,
                bias_spec=[
                    {
                        "module_name": "rsisignal",
                        "timeframes": ["D"],
                        "params": params,
                    }
                    for params in combo_params
                ],
            ),
        ),
        evaluation_defaults=EvaluationDefaultsCatalog(
            continuous=EvaluationPhaseDefaultsConfig(bias_spec=eval_bias_spec),
            signed_signal=EvaluationPhaseDefaultsConfig(bias_spec=eval_bias_spec),
        ),
        validation_robustness=replace(
            base.validation_robustness,
            n_bootstrap=100,
            random_seed=7,
            rolling_window=20,
        ),
    )


def test_run_validation_robustness_pipeline_builds_core_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    target = pd.Series(
        [0.01 if index % 5 else -0.005 for index in range(len(index))],
        index=index,
        dtype=float,
        name="target",
    )
    combo_params = [{"lookback": 5}, {"lookback": 20}, {"lookback": 50}]
    enriched_combo_params = [
        enrich_param_combo_with_module(params, "rsisignal") for params in combo_params
    ]
    signal_frames = {
        combo_key(params): _build_combo_frame(
            index,
            target,
            {
                5: 1.0,
                20: 0.35,
                50: -0.6,
            }[params["lookback"]],
            signal_name=f"sig_{params['lookback']}",
            alternating=params["lookback"] in {5, 50},
        )
        for params in enriched_combo_params
    }

    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsisignal", "timeframes": ["D"], "params": params}
            for params in combo_params
        ],
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.load_signed_signal_research_data",
        lambda _config, _expanded, print_loaded=False: SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=enriched_combo_params,
            reference_index=index,
            reference_target_series=target,
        ),
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner._load_is_metrics_from_csv",
        lambda _config, _metric_column: None,
    )

    config = _validation_test_config(
        combo_params=combo_params,
        eval_params={"lookback": 5},
        index=index,
    )
    report, rank_rows, chosen_label = run_validation_robustness_pipeline(config)

    assert chosen_label == research_display_label(enriched_combo_params[0])
    assert len(rank_rows) == 3
    assert report.sharpe_comparison.sr_is is not None
    assert report.cusum.n_obs > 0
    assert len(report.equity_curve_bands.actual) == report.cusum.n_obs
    assert report.rank_correlation.n_combinations == 3
    assert isinstance(report.all_passed, bool)
    assert report.interpretation


def test_run_validation_robustness_pipeline_skips_rank_correlation_for_two_combos(
    monkeypatch,
) -> None:
    """Two-point grids (e.g. sma 200 vs 252) must not call Spearman (non-finite p-value)."""
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    target = pd.Series(
        [0.01 if position % 5 else -0.005 for position in range(len(index))],
        index=index,
        dtype=float,
        name="target",
    )
    combo_params = [{"sma_period": 200}, {"sma_period": 252}]
    enriched = [
        enrich_param_combo_with_module(p, "seasonalindiceseof") for p in combo_params
    ]
    signal_frames = {
        combo_key(p): _build_combo_frame(index, target, 1.0, signal_name="signal")
        for p in enriched
    }

    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.populate_cache_if_needed",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "seasonal_indices_eof", "timeframes": ["D"], "params": p}
            for p in combo_params
        ],
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.load_signed_signal_research_data",
        lambda _config, _expanded, print_loaded=False: SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=enriched,
            reference_index=index,
            reference_target_series=target,
        ),
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner._load_is_metrics_from_csv",
        lambda _config, _metric_column: None,
    )

    config = _validation_test_config(
        combo_params=combo_params,
        eval_params={"sma_period": 200},
        index=index,
    )
    config = replace(
        config,
        evaluation_defaults=EvaluationDefaultsCatalog(
            continuous=EvaluationPhaseDefaultsConfig(
                bias_spec={
                    "module_name": "seasonal_indices_eof",
                    "timeframes": ["D"],
                    "params": {"sma_period": 200},
                }
            ),
            signed_signal=EvaluationPhaseDefaultsConfig(
                bias_spec={
                    "module_name": "seasonal_indices_eof",
                    "timeframes": ["D"],
                    "params": {"sma_period": 200},
                }
            ),
        ),
    )
    report, rank_rows, _ = run_validation_robustness_pipeline(config)
    assert len(rank_rows) == 2
    assert "not applicable" in report.rank_correlation.interpretation.lower()


def test_run_validation_robustness_pipeline_skips_rank_correlation_for_single_combo(
    monkeypatch,
) -> None:
    """Rule-based specs with no param grid (e.g. eoy_sp500) must not require rank correlation."""
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    target = pd.Series(
        [0.01 if position % 5 else -0.005 for position in range(len(index))],
        index=index,
        dtype=float,
        name="target",
    )
    combo_params = [{"mode": "tue_wed"}]
    enriched_combo_params = [
        enrich_param_combo_with_module(params, "turnaroundtuesday") for params in combo_params
    ]
    signal_frames = {
        combo_key(params): _build_combo_frame(
            index,
            target,
            1.0,
            signal_name="signal",
        )
        for params in enriched_combo_params
    }

    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.populate_cache_if_needed",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "turnaroundtuesday", "timeframes": ["D"], "params": params}
            for params in combo_params
        ],
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.load_signed_signal_research_data",
        lambda _config, _expanded, print_loaded=False: SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=enriched_combo_params,
            reference_index=index,
            reference_target_series=target,
        ),
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner._load_is_metrics_from_csv",
        lambda _config, _metric_column: None,
    )

    config = _validation_test_config(
        combo_params=combo_params,
        eval_params={"mode": "tue_wed"},
        index=index,
    )
    report, rank_rows, chosen_label = run_validation_robustness_pipeline(config)

    assert len(rank_rows) == 1
    assert rank_rows[0].is_chosen
    assert chosen_label == research_display_label(enriched_combo_params[0])
    assert "not applicable" in report.rank_correlation.interpretation.lower()
    assert report.rank_correlation.passed is True
    assert report.sharpe_comparison.sr_is is not None


def test_write_validation_robustness_summary_persists_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    index = pd.date_range("2021-01-01", periods=200, freq="D")
    target = pd.Series(
        [0.01 + 0.002 * (index % 7) - 0.001 * (index % 3) for index in range(len(index))],
        index=index,
        dtype=float,
        name="target",
    )
    combo_params = [{"lookback": 5}, {"lookback": 20}, {"lookback": 50}]
    enriched_combo_params = [
        enrich_param_combo_with_module(params, "rsisignal") for params in combo_params
    ]
    signal_frames = {
        combo_key(params): _build_combo_frame(
            index,
            target,
            {
                5: 0.8,
                20: 0.2,
                50: -0.5,
            }[params["lookback"]],
            signal_name=f"sig_{params['lookback']}",
            alternating=params["lookback"] != 20,
        )
        for params in enriched_combo_params
    }

    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsisignal", "timeframes": ["D"], "params": params}
            for params in combo_params
        ],
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.load_signed_signal_research_data",
        lambda _config, _expanded, print_loaded=False: SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=enriched_combo_params,
            reference_index=index,
            reference_target_series=target,
        ),
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner._load_is_metrics_from_csv",
        lambda _config, _metric_column: None,
    )

    config = _validation_test_config(
        combo_params=combo_params,
        eval_params={"lookback": 20},
        index=index,
    )
    report, rank_rows, chosen_label = run_validation_robustness_pipeline(config)
    artifacts = write_validation_robustness_summary(
        report,
        tmp_path,
        rank_scatter_rows=rank_rows,
        chosen_combo_label=chosen_label,
    )

    assert artifacts["json"].exists()
    assert artifacts["rank_correlation_csv"].exists()
    assert artifacts["z_cusum_sr_csv"].exists()
    assert artifacts["equity_bands_csv"].exists()
    reloaded = load_validation_robustness_report_from_json(artifacts["json"])
    assert reloaded.all_passed == report.all_passed
    assert reloaded.rank_correlation.spearman_rho == report.rank_correlation.spearman_rho


def test_run_and_write_validation_robustness_writes_plots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    index = pd.date_range("2022-01-01", periods=200, freq="D")
    target = pd.Series(
        [0.015 * (1 + 0.1 * (index % 11)) if index % 3 else -0.004 for index in range(len(index))],
        index=index,
        dtype=float,
        name="target",
    )
    combo_params = [{"lookback": 5}, {"lookback": 20}, {"lookback": 50}]
    enriched_combo_params = [
        enrich_param_combo_with_module(params, "rsisignal") for params in combo_params
    ]
    signal_frames = {
        combo_key(params): _build_combo_frame(
            index,
            target,
            {
                5: 1.0,
                20: 0.3,
                50: -0.7,
            }[params["lookback"]],
            signal_name=f"sig_{params['lookback']}",
            alternating=params["lookback"] in {5, 50},
        )
        for params in enriched_combo_params
    }

    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsisignal", "timeframes": ["D"], "params": params}
            for params in combo_params
        ],
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.load_signed_signal_research_data",
        lambda _config, _expanded, print_loaded=False: SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=enriched_combo_params,
            reference_index=index,
            reference_target_series=target,
        ),
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner._load_is_metrics_from_csv",
        lambda _config, _metric_column: None,
    )

    config = _validation_test_config(
        combo_params=combo_params,
        eval_params={"lookback": 5},
        index=index,
    )
    artifacts = run_and_write_validation_robustness(config, output_dir=tmp_path)

    assert (tmp_path / "validation_robustness_report.json").exists()
    assert (tmp_path / "matplotlib" / "validation_sharpe_comparison.png").exists()
    assert (tmp_path / "matplotlib" / "validation_z_cusum_rolling_sr.png").exists()
    assert (tmp_path / "matplotlib" / "validation_equity_bands.png").exists()
    assert (tmp_path / "matplotlib" / "validation_rank_scatter.png").exists()
    assert artifacts["json"] == tmp_path / "validation_robustness_report.json"


def test_params_match_links_filter_gate_eval_to_is_signal_combo() -> None:
    signal_params = {
        "period": 10,
        "std_dev": 2.5,
        "lower_threshold": 0.0,
        "upper_threshold": 0.5,
        "strategy_mode": "long",
        "exit_policy": "threshold",
        "exit_bars": 5,
    }
    gate = enrich_param_combo_with_module(
        {
            "filter_module": "sma_above_filter",
            "filter_params": {"period": 252},
            "signal_module": "percent_b_signal",
            "signal_params": signal_params,
        },
        "filter_gate_entry_only",
    )
    plain = enrich_param_combo_with_module(signal_params, "percent_b_signal")
    assert _params_match(gate, plain)
    rank_chosen = _rank_scatter_chosen_params(gate)
    assert combo_key(rank_chosen) == combo_key(plain)


def test_resolve_chosen_params_matches_eval_in_exploration_grid() -> None:
    config = load_config()
    chosen = _resolve_chosen_params(config, None)
    eval_spec = config.eval_bias_spec
    eval_module = str(eval_spec.get("module_name", ""))
    eval_params = dict(eval_spec.get("params", {}))

    if eval_module in {"filter_gate", "filter_gate_entry_only"}:
        signal_module = str(eval_params.get("signal_module", ""))
        assert signal_module == "rsi_signal"
        signal_params = dict(eval_params.get("signal_params", {}))
        assert chosen.get("_bias_module") == eval_module
        nested = chosen.get("signal_params")
        assert isinstance(nested, dict)
        assert combo_key(nested) == combo_key(signal_params)
        assert signal_params["rsi_period"] == 2
        assert signal_params["oversold"] == 25.0
        assert signal_params["overbought"] == 75.0
        eval_key = combo_key(
            enrich_param_combo_with_module(signal_params, signal_module),
        )
    else:
        assert chosen.get("_bias_module") == eval_module
        eval_key = combo_key(chosen)

    grid_keys = {
        combo_key(
            enrich_param_combo_with_module(spec["params"], spec.get("module_name")),
        )
        for spec in expand_bias_specs(config.bias_spec)
    }
    assert eval_key in grid_keys


def test_run_validation_robustness_reuses_preloaded_eval_data(
    monkeypatch,
) -> None:
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    target = pd.Series(0.001, index=index, dtype=float, name="target")
    combo_params = [{"lookback": 5}, {"lookback": 20}, {"lookback": 50}]
    enriched_combo_params = [
        enrich_param_combo_with_module(params, "rsisignal") for params in combo_params
    ]
    signal_frames = {
        combo_key(params): _build_combo_frame(
            index,
            target,
            {
                5: 0.5,
                20: -0.2,
                50: 0.1,
            }[params["lookback"]],
            signal_name=f"sig_{params['lookback']}",
        )
        for params in enriched_combo_params
    }
    eval_params = enrich_param_combo_with_module({"lookback": 5}, "rsisignal")
    eval_frame = signal_frames[combo_key(eval_params)]
    eval_data = SimpleNamespace(
        combo_signal_target={combo_key(eval_params): eval_frame},
        successful_param_grid=[eval_params],
        reference_index=index,
        reference_target_series=target,
    )
    load_call_counts: list[int] = []

    def _mock_load(_config: object, expanded: list[dict[str, object]], print_loaded: bool = False) -> SimpleNamespace:
        load_call_counts.append(len(expanded))
        return SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=enriched_combo_params,
            reference_index=index,
            reference_target_series=target,
        )

    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.populate_cache_if_needed",
        lambda _config, **kwargs: None,
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.expand_bias_specs",
        lambda bias_spec: [
            {"module_name": "rsisignal", "timeframes": ["D"], "params": params}
            for params in combo_params
        ],
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner.load_signed_signal_research_data",
        _mock_load,
    )
    monkeypatch.setattr(
        "feature_research.validation.robustness_runner._load_is_metrics_from_csv",
        lambda _config, _metric_column: None,
    )

    config = _validation_test_config(
        combo_params=combo_params,
        eval_params={"lookback": 5},
        index=index,
    )
    run_validation_robustness_pipeline(
        config,
        eval_research_data=eval_data,
    )

    assert load_call_counts == [len(combo_params)]


def test_apply_fast_validation_profile_disables_gate() -> None:
    from feature_research.config import apply_fast_validation_profile

    fast = apply_fast_validation_profile(load_config())
    assert fast.portfolio_addition_gate.enabled is False
    assert fast.validation_robustness.n_bootstrap == 200
