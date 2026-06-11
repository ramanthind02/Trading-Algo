from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import json

import numpy as np
import pandas as pd
import pytest
from quantfoundry_core.portfolio_gate import compute_portfolio_addition_gate

from research.feature.config import load_config, resolve_portfolio_gate_n_jobs
from lib.core.enums import Ticker
from research.feature.config import PortfolioAdditionGateConfig
from research.feature.portfolio_addition.gate_runner import (
    PortfolioGateInputs,
    _GatePhaseTask,
    _align_series_to_arrays,
    _analytical_hurdle_for_sleeve,
    _compute_first_in_sleeve_gate,
    _run_gate_phase_tasks,
    _write_sleeve_tearsheet_artifacts,
    block_bootstrap_delta_sr_samples,
    build_portfolio_config_for_gate,
    compute_dual_scope_portfolio_addition_gate,
    load_portfolio_addition_report_from_json,
    run_and_write_portfolio_addition_gate,
    write_portfolio_addition_summary,
)
from research.portfolio.config import PortfolioResearchConfig
from research.portfolio.pipelines.portfolio_test import PhaseResult


def test_canonical_portfolio_gate_defaults_to_serial() -> None:
    """C1 regression: the portfolio-addition gate must default to n_jobs=1.

    loky workers do NOT inherit the process-global research feed / EWSD blend, so a
    parallel gate silently fits ensembles under the default "futures" feed while the
    eval ran under the configured feed (e.g. cfd) -> corrupts dSR / pass-fail. The
    canonical/CLI path (load_config) must therefore resolve the gate to serial.
    """
    config = load_config()
    assert config.portfolio_addition_gate.n_jobs is None  # inherit -> resolver decides
    assert resolve_portfolio_gate_n_jobs(config) == 1
    # An explicit override is still honoured (opt-in; caller owns the feed caveat).
    parallel = replace(
        config,
        portfolio_addition_gate=replace(config.portfolio_addition_gate, n_jobs=4),
    )
    assert resolve_portfolio_gate_n_jobs(parallel) == 4


def _synthetic_inputs(*, n: int = 260, seed: int = 11) -> PortfolioGateInputs:
    rng = np.random.default_rng(seed)
    existing_a = rng.normal(0.0004, 0.01, size=n)
    existing_b = rng.normal(0.0003, 0.011, size=n)
    new_strategy = rng.normal(0.0005, 0.012, size=n)
    portfolio_without = 0.5 * existing_a + 0.5 * existing_b
    portfolio_with = 0.4 * existing_a + 0.35 * existing_b + 0.25 * new_strategy
    index = pd.date_range("2018-01-01", periods=n, freq="D")
    labels = ["candidate", "ens_a", "ens_b"]
    matrix = np.array(
        [
            [1.0, float(np.corrcoef(new_strategy, existing_a)[0, 1]), float(np.corrcoef(new_strategy, existing_b)[0, 1])],
            [float(np.corrcoef(existing_a, new_strategy)[0, 1]), 1.0, float(np.corrcoef(existing_a, existing_b)[0, 1])],
            [float(np.corrcoef(existing_b, new_strategy)[0, 1]), float(np.corrcoef(existing_b, existing_a)[0, 1]), 1.0],
        ]
    )
    weight_without = pd.DataFrame(
        {
            "weight_layer_ticker": ["__GLOBAL__", "__GLOBAL__"],
            "stream_or_model_id": ["s1", "s2"],
            "stream_source_model_name": ["ens_a", "ens_b"],
            "stream_weight": [0.55, 0.45],
            "weight_method": ["ledoit_wolf_min_corr", "ledoit_wolf_min_corr"],
        }
    )
    weight_with = pd.DataFrame(
        {
            "weight_layer_ticker": ["__GLOBAL__", "__GLOBAL__", "__GLOBAL__"],
            "stream_or_model_id": ["s1", "s2", "s3"],
            "stream_source_model_name": ["ens_a", "ens_b", "candidate"],
            "stream_weight": [0.40, 0.35, 0.25],
            "weight_method": ["ledoit_wolf_min_corr", "ledoit_wolf_min_corr", "ledoit_wolf_min_corr"],
        }
    )
    return PortfolioGateInputs(
        new_strategy_returns=new_strategy,
        existing_strategy_returns={"ens_a": existing_a, "ens_b": existing_b},
        portfolio_returns_without=portfolio_without,
        portfolio_returns_with=portfolio_with,
        weight_assigned=0.25,
        candidate_key="candidate",
        eval_start=pd.Timestamp(index[0]),
        eval_end=pd.Timestamp(index[-1]),
        weight_layer_without_df=weight_without,
        weight_layer_with_df=weight_with,
        pairwise_corr_matrix=pd.DataFrame(matrix, index=labels, columns=labels),
    )


def test_compute_portfolio_addition_gate_from_synthetic_inputs() -> None:
    inputs = _synthetic_inputs()
    report = compute_portfolio_addition_gate(
        inputs.new_strategy_returns,
        inputs.existing_strategy_returns,
        inputs.portfolio_returns_without,
        inputs.portfolio_returns_with,
        weight_assigned=inputs.weight_assigned,
        n_bootstrap=200,
        bootstrap_seed=3,
    )
    assert report.empirical_comparison.delta_sr == report.empirical_comparison.sr_with - report.empirical_comparison.sr_without
    assert isinstance(report.passed, bool)
    assert report.interpretation
    assert report.risk_impact is not None
    assert report.passed == (report.empirical_comparison.passed and report.risk_impact.passed)
    assert "risk_impact" in report.to_json_dict()
    assert "gate_criteria" in report.to_json_dict()


def test_write_and_reload_portfolio_addition_summary(tmp_path: Path) -> None:
    inputs = _synthetic_inputs()
    report = compute_portfolio_addition_gate(
        inputs.new_strategy_returns,
        inputs.existing_strategy_returns,
        inputs.portfolio_returns_without,
        inputs.portfolio_returns_with,
        weight_assigned=inputs.weight_assigned,
        n_bootstrap=100,
        bootstrap_seed=5,
    )
    samples, _ = block_bootstrap_delta_sr_samples(
        inputs.portfolio_returns_without,
        inputs.portfolio_returns_with,
        n_bootstrap=100,
        seed=5,
    )
    payload = {
        "skipped": False,
        "meta": {
            "candidate_key": "candidate",
            "eval_start": "2018-01-01",
            "eval_end": "2018-09-16",
        },
        **report.to_json_dict(),
    }
    artifacts = write_portfolio_addition_summary(
        payload,
        tmp_path,
        bootstrap_samples=samples.tolist(),
    )
    assert artifacts["json"].exists()
    reloaded = load_portfolio_addition_report_from_json(artifacts["json"])
    assert reloaded is not None
    assert reloaded.passed == report.passed
    assert reloaded.empirical_comparison.delta_sr == report.empirical_comparison.delta_sr
    assert reloaded.risk_impact.delta_max_dd == report.risk_impact.delta_max_dd


def test_composite_pass_false_when_risk_fails_despite_sr() -> None:
    inputs = _synthetic_inputs()
    portfolio_with = inputs.portfolio_returns_with.copy()
    portfolio_with[-30:] -= 0.06
    report = compute_portfolio_addition_gate(
        inputs.new_strategy_returns,
        inputs.existing_strategy_returns,
        inputs.portfolio_returns_without,
        portfolio_with,
        weight_assigned=inputs.weight_assigned,
        n_bootstrap=100,
        bootstrap_seed=7,
        delta_sr_threshold=-10.0,
        max_dd_tolerance=0.001,
    )
    assert report.empirical_comparison.passed
    assert not report.risk_impact.passed
    assert not report.passed


def test_gc_only_portfolio_discovery_includes_gc_breakout_only() -> None:
    from research.feature.config import (
        PortfolioSourceConfig,
        discover_portfolio_source_ensemble_dirs,
    )

    source = PortfolioSourceConfig(vault_profile="prop")
    discovered = discover_portfolio_source_ensemble_dirs(
        source,
        portfolio_tickers=(Ticker.GC,),
    )
    assert set(discovered) == {"robust_trend_breakout_gc_long"}


def test_build_portfolio_config_for_gate_allows_empty_baseline_for_gc_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from research.feature.config import PortfolioSourceConfig
    from research.feature.ui.planner import apply_ui_request, build_ui_request
    from research.feature.shared import FeatureResearchPhase
    from research.portfolio.config import EnsembleDirsPolicy

    config = load_config()
    request = build_ui_request(
        phase_name=FeatureResearchPhase.VALIDATION.value,
        tickers_text="GC",
        fallback_tickers=tuple(config.tickers),
    )
    configured = apply_ui_request(config, request)
    configured = replace(
        configured,
        tickers=(Ticker.GC,),
        portfolio_source=replace(
            configured.portfolio_source or PortfolioSourceConfig(),
            tickers=(Ticker.ES, Ticker.NQ, Ticker.GC),
        ),
        portfolio_inclusion=replace(
            configured.portfolio_inclusion,
            candidate_tickers=(Ticker.GC,),
        ),
    )
    monkeypatch.setattr(
        "research.feature.portfolio_addition.gate_runner._baseline_ensemble_dirs_for_gate",
        lambda _research: {},
    )
    portfolio_config = build_portfolio_config_for_gate(configured)
    assert portfolio_config.tickers == (Ticker.ES, Ticker.GC, Ticker.NQ)
    assert portfolio_config.ensemble_dirs == {}
    assert portfolio_config.ensemble_dirs_policy is EnsembleDirsPolicy.ALLOW_EMPTY


def test_build_portfolio_config_for_gate_uses_research_window() -> None:
    config = load_config()
    portfolio_config = build_portfolio_config_for_gate(config)
    assert config.research_window is not None
    assert portfolio_config.train_window.start == config.research_window.train_start
    assert portfolio_config.validation_window.end == config.research_window.val_end
    assert portfolio_config.weight_layer_method == config.portfolio_source.weight_layer_method
    assert portfolio_config.benchmark_ticker is None
    assert portfolio_config.baseline_mode == "equal_weight"
    if portfolio_config.weight_layer_method == "hierarchy_equal":
        assert portfolio_config.weight_layer_kwargs.get("hierarchy_spec")


def test_build_portfolio_config_for_gate_keeps_multi_asset_candidate_tickers() -> None:
    """Baseline vault scoping must not drop CL/FX legs from the inclusion candidate."""
    config = load_config()
    candidate_names = {
        ticker.name for ticker in (config.portfolio_inclusion.candidate_tickers or ())
    }
    if len(candidate_names) <= 3:
        pytest.skip("load_config() has no multi-asset portfolio_inclusion.candidate_tickers")

    portfolio_config = build_portfolio_config_for_gate(config)
    portfolio_names = {ticker.name for ticker in portfolio_config.tickers}
    assert candidate_names <= portfolio_names


def test_align_series_to_arrays_intersects_sleeve_and_portfolio_lengths() -> None:
    index_full = pd.date_range("2020-01-01", periods=100, freq="B")
    index_short = index_full[10:]
    aligned = _align_series_to_arrays(
        {
            "new": pd.Series(0.001, index=index_short),
            "portfolio_without": pd.Series(0.002, index=index_full),
            "sleeve_without": pd.Series(0.003, index=index_full),
            "sleeve_with": pd.Series(0.004, index=index_short[5:]),
        }
    )
    expected_len = len(index_full.intersection(index_short).intersection(index_short[5:]))
    assert len(aligned["new"]) == expected_len
    assert len(aligned["sleeve_without"]) == expected_len
    assert len(aligned["portfolio_without"]) == expected_len


def test_analytical_hurdle_for_empty_sleeve_respects_passed_contract() -> None:
    """Degenerate sleeve baseline: passed must equal sr_new > hurdle (not always True)."""
    flat = np.zeros(200, dtype=float)
    negative = np.full(200, -0.001, dtype=float)
    hurdle = _analytical_hurdle_for_sleeve(negative, flat)
    assert hurdle.hurdle == 0.0
    assert hurdle.passed == (hurdle.sr_new > hurdle.hurdle)
    assert hurdle.passed is False

    positive = np.full(200, 0.001, dtype=float)
    hurdle_pos = _analytical_hurdle_for_sleeve(positive, flat)
    assert hurdle_pos.passed == (hurdle_pos.sr_new > hurdle_pos.hurdle)
    assert hurdle_pos.passed is True


def test_first_in_sleeve_skips_risk_fail_against_flat_baseline() -> None:
    n = 300
    rng = np.random.default_rng(0)
    new_returns = rng.normal(0.001, 0.01, size=n)
    flat = np.zeros(n, dtype=float)
    report = _compute_first_in_sleeve_gate(
        new_returns,
        flat,
        new_returns,
        weight_assigned=1.0,
        pairwise_corr_flag_threshold=0.75,
        delta_sr_threshold=0.02,
        max_dd_tolerance=0.01,
        max_ulcer_increase=0.05,
        stress_max_dd_tolerance=0.01,
        weight_floor=0.05,
        n_bootstrap=100,
        bootstrap_seed=1,
    )
    assert report.passed == report.empirical_comparison.passed


def test_dual_scope_first_in_sleeve_avoids_nan_pairwise() -> None:
    base = _synthetic_inputs()
    inputs = replace(
        base,
        sleeve_returns_without=base.portfolio_returns_without,
        sleeve_returns_with=base.new_strategy_returns,
        sleeve_existing_strategy_returns={},
        sleeve_weight_assigned=1.0,
        sleeve_asset_class="equity_indices",
        sleeve_style_group="mean_reversion_indices",
    )
    primary, _portfolio = compute_dual_scope_portfolio_addition_gate(
        inputs,
        gate_cfg=PortfolioAdditionGateConfig(enabled=True, n_bootstrap=100),
    )
    assert np.isfinite(primary.pairwise_redundancy.max_pairwise_corr)


def test_dual_scope_weight_warning_follows_sleeve_assessment_only() -> None:
    """Primary report uses sleeve weight_assessment; must not OR portfolio warning."""
    base = _synthetic_inputs()
    ens_a = base.existing_strategy_returns["ens_a"]
    new_strategy = base.new_strategy_returns
    inputs = replace(
        base,
        sleeve_returns_without=ens_a.copy(),
        sleeve_returns_with=0.5 * ens_a + 0.5 * new_strategy,
        sleeve_existing_strategy_returns={"ens_a": ens_a},
        sleeve_weight_assigned=0.20,
        weight_assigned=0.01,
        sleeve_asset_class="equity_indices",
        sleeve_style_group="mean_reversion_indices",
    )
    primary, portfolio = compute_dual_scope_portfolio_addition_gate(
        inputs,
        gate_cfg=PortfolioAdditionGateConfig(enabled=True, n_bootstrap=100, weight_floor=0.05),
    )
    assert primary.weight_warning is (not primary.weight_assessment.meaningful)
    assert primary.weight_assessment.meaningful is True
    assert portfolio.weight_warning is True


def test_build_portfolio_config_for_gate_buy_hold_uses_portfolio_ticker_when_es_missing() -> None:
    config = replace(
        load_config(),
        portfolio_source=replace(
            load_config().portfolio_source,
            tickers=(Ticker.NQ,),
            baseline_mode="buy_hold",
            benchmark_ticker=None,
        ),
    )
    portfolio_config = build_portfolio_config_for_gate(config)
    assert portfolio_config.benchmark_ticker == Ticker.NQ


def test_write_sleeve_tearsheet_artifacts_respects_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs()
    phase = PhaseResult(
        name="sleeve",
        output_dir=tmp_path,
        combined_strategy_returns=pd.Series(0.001, index=pd.date_range("2020-01-01", periods=50, freq="B")),
        combined_baseline_returns=pd.Series(0.0, index=pd.date_range("2020-01-01", periods=50, freq="B")),
    )
    inputs_with_phases = replace(
        inputs,
        sleeve_asset_class="equity_indices",
        sleeve_style_group="mean_reversion_indices",
        sleeve_phase_with_tr=phase,
        sleeve_phase_with_val=phase,
    )
    gate_cfg_off = PortfolioAdditionGateConfig(enabled=True, emit_sleeve_tearsheets=False)
    assert _write_sleeve_tearsheet_artifacts(
        output_dir=tmp_path,
        inputs=inputs_with_phases,
        gate_cfg=gate_cfg_off,
    ) == ()

    calls: list[int] = []

    def _fake_write(**kwargs: object) -> tuple[Path, ...]:
        calls.append(1)
        return (tmp_path / "sleeve_tearsheets_equity_indices_mean_reversion_indices" / "x.html",)

    monkeypatch.setattr(
        "research.feature.portfolio_addition.gate_runner.write_sleeve_level_tearsheets",
        _fake_write,
    )
    gate_cfg_on = PortfolioAdditionGateConfig(enabled=True, emit_sleeve_tearsheets=True)
    paths = _write_sleeve_tearsheet_artifacts(
        output_dir=tmp_path,
        inputs=inputs_with_phases,
        gate_cfg=gate_cfg_on,
    )
    assert len(paths) == 1
    assert calls == [1]


def test_run_and_write_portfolio_addition_gate_with_monkeypatched_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = replace(load_config(), output_root=tmp_path)

    def _fake_pipeline(research):  # noqa: ANN001
        inputs = _synthetic_inputs()
        report = compute_portfolio_addition_gate(
            inputs.new_strategy_returns,
            inputs.existing_strategy_returns,
            inputs.portfolio_returns_without,
            inputs.portfolio_returns_with,
            weight_assigned=inputs.weight_assigned,
            n_bootstrap=100,
            bootstrap_seed=9,
        )
        return report, report, inputs, "candidate", None

    monkeypatch.setattr(
        "research.feature.portfolio_addition.gate_runner.run_portfolio_addition_gate_pipeline",
        _fake_pipeline,
    )
    artifacts = run_and_write_portfolio_addition_gate(config, output_dir=tmp_path)
    assert (tmp_path / "portfolio_addition_report.json").exists()
    assert (tmp_path / "portfolio_addition_delta_sr_bootstrap.csv").exists()
    assert (tmp_path / "portfolio_addition_pairwise_corr_matrix.csv").exists()
    assert (tmp_path / "portfolio_addition_weight_layer_members.csv").exists()
    assert (tmp_path / "portfolio_addition_drawdown_detail.csv").exists()
    assert (tmp_path / "portfolio_risk_impact.csv").exists()
    assert (tmp_path / "matplotlib" / "portfolio_addition_delta_sr_histogram.png").exists()
    assert (tmp_path / "matplotlib" / "portfolio_addition_correlation_heatmap.png").exists()
    assert (tmp_path / "matplotlib" / "portfolio_addition_drawdown_correlation.png").exists()
    assert (tmp_path / "matplotlib" / "portfolio_addition_risk_impact_deltas.png").exists()
    payload = json.loads((tmp_path / "portfolio_addition_report.json").read_text(encoding="utf-8"))
    assert "context" in payload
    assert "pairwise_peers" in payload["context"]
    assert "risk_impact" in payload
    assert "gate_criteria" in payload
    assert artifacts["json"].exists()


def test_run_gate_phase_tasks_invokes_all_phases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def _fake_phase(
        _config: PortfolioResearchConfig,
        phase: str,
        *,
        emit_tearsheets: bool = False,
        run_preflight: bool = True,
        run_purpose: str = "full",
    ) -> PhaseResult:
        calls.append(phase)
        index = pd.date_range("2020-01-01", periods=30, freq="B")
        returns = pd.Series(0.001, index=index)
        return PhaseResult(
            name=phase,
            output_dir=tmp_path,
            combined_strategy_returns=returns,
            combined_baseline_returns=returns * 0.0,
            weight_layer_diagnostics_df=pd.DataFrame(),
        )

    monkeypatch.setattr(
        "research.feature.portfolio_addition.gate_runner.run_single_phase_for_prop_firm",
        _fake_phase,
    )
    portfolio_config = build_portfolio_config_for_gate(load_config())
    tasks = [
        _GatePhaseTask("without_tr", portfolio_config, "train"),
        _GatePhaseTask("without_val", portfolio_config, "validation"),
    ]
    results = _run_gate_phase_tasks(tasks, n_jobs=1)
    assert set(calls) == {"train", "validation"}
    assert set(results) == {"without_tr", "without_val"}
