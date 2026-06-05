"""Portfolio addition gate orchestration (Quant Foundry Core)."""
from __future__ import annotations

import dataclasses
import json
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
import pandas as pd
from quantfoundry_core.portfolio_gate import PortfolioAdditionReport, compute_portfolio_addition_gate
from quantfoundry_core.portfolio_gate.models import (
    AnalyticalHurdleResult,
    BootstrapCI,
    EmpiricalComparisonResult,
    IDMImprovementResult,
    PairwiseRedundancyResult,
    PortfolioRiskImpactResult,
    WeightAssessmentResult,
)
from quantfoundry_core.portfolio_gate.risk_impact import compute_portfolio_risk_impact

from research.feature.config import (
    PortfolioSourceConfig,
    ResearchConfig,
    discover_portfolio_source_ensemble_dirs,
    resolve_portfolio_benchmark_ticker,
    resolve_portfolio_gate_n_jobs,
)
from ensemble.vault.hierarchy_spec import _asset_class_for_stream
from research.feature.inclusion_gates import (
    _concat_returns,
    _returns_suitable_for_quantstats_tearsheet,
    config_with_ensemble_dirs,
    infer_candidate_key,
    materialize_inclusion_candidate_from_eval_bias_spec,
    write_candidate_strategy_tearsheets,
    write_full_portfolio_comparison_tearsheets,
    write_sleeve_level_tearsheets,
)
from research.feature.config import PortfolioAdditionGateConfig
from research.feature.portfolio_addition.sleeve_gate import (
    candidate_sleeve_identity,
    candidate_stream_ids_for_ensemble,
    candidate_ticker_names_from_ensemble,
    ensemble_dirs_in_sleeve,
    sleeve_label,
)
from quantfoundry_core.portfolio_gate.empirical import compute_empirical_comparison
from quantfoundry_core.portfolio_gate.hurdle import compute_analytical_hurdle
from quantfoundry_core.portfolio_gate.idm import compute_idm_improvement
from quantfoundry_core.portfolio_gate.weight import compute_weight_assessment
from research.portfolio.config import (
    EnsembleDirsPolicy,
    PortfolioResearchConfig,
    ResearchWindow,
    load_config as load_portfolio_config,
)
from research.portfolio.pipelines.portfolio_test import (
    PhaseResult,
    run_portfolio_research_cache_preflight,
    run_single_phase_for_prop_firm,
)
from research.portfolio.weight_layer_export import _GLOBAL_SYNTHETIC_TICKER
from lib.core.enums import Ticker

_DataclassT = TypeVar("_DataclassT")
_PERIODS_PER_YEAR = 252
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PortfolioGateInputs:
    """Aligned return arrays and weight for Core portfolio gate."""

    new_strategy_returns: np.ndarray
    existing_strategy_returns: dict[str, np.ndarray]
    portfolio_returns_without: np.ndarray
    portfolio_returns_with: np.ndarray
    weight_assigned: float
    candidate_key: str
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    weight_layer_without_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    weight_layer_with_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairwise_corr_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_ensemble_dirs: dict[str, str] = field(default_factory=dict)
    merged_ensemble_dirs: dict[str, str] = field(default_factory=dict)
    portfolio_ticker_names: frozenset[str] = field(default_factory=frozenset)
    sleeve_asset_class: str = ""
    sleeve_style_group: str = ""
    sleeve_baseline_ensemble_dirs: dict[str, str] = field(default_factory=dict)
    sleeve_returns_without: np.ndarray = field(default_factory=lambda: np.array([]))
    sleeve_returns_with: np.ndarray = field(default_factory=lambda: np.array([]))
    sleeve_existing_strategy_returns: dict[str, np.ndarray] = field(default_factory=dict)
    sleeve_weight_assigned: float = 0.0
    candidate_stream_ids: frozenset[str] = field(default_factory=frozenset)
    hierarchy_spec_without: dict[str, object] | None = None
    hierarchy_spec_with: dict[str, object] | None = None
    sleeve_phase_without_tr: PhaseResult | None = None
    sleeve_phase_without_val: PhaseResult | None = None
    sleeve_phase_with_tr: PhaseResult | None = None
    sleeve_phase_with_val: PhaseResult | None = None
    portfolio_phase_without_tr: PhaseResult | None = None
    portfolio_phase_without_val: PhaseResult | None = None
    portfolio_phase_with_tr: PhaseResult | None = None
    portfolio_phase_with_val: PhaseResult | None = None
    phase_candidate_tr: PhaseResult | None = None
    phase_candidate_val: PhaseResult | None = None


def _baseline_ensemble_dirs_for_gate(research: ResearchConfig) -> dict[str, str]:
    source = research.portfolio_source or PortfolioSourceConfig()
    tickers = list(source.tickers) if source.tickers is not None else list(research.tickers)
    return discover_portfolio_source_ensemble_dirs(source, portfolio_tickers=tickers)


def _union_portfolio_tickers(*ticker_groups: Iterable[Ticker]) -> tuple[Ticker, ...]:
    """Merge ticker iterables, preserving one ``Ticker`` per symbol name."""
    merged: dict[str, Ticker] = {}
    for group in ticker_groups:
        for ticker in group:
            key = ticker.name if hasattr(ticker, "name") else str(ticker)
            merged[key] = ticker
    return tuple(sorted(merged.values(), key=lambda item: item.name))


def build_portfolio_config_for_gate(research: ResearchConfig) -> PortfolioResearchConfig:
    """Build a portfolio research config aligned with feature-research validation windows."""

    base = load_portfolio_config()
    source = research.portfolio_source or PortfolioSourceConfig()
    if research.research_window is None:
        raise ValueError("research_window is required for the portfolio addition gate")

    ensemble_dirs = _baseline_ensemble_dirs_for_gate(research)
    inclusion = research.portfolio_inclusion
    candidate_tickers = tuple(inclusion.candidate_tickers or ())
    base_tickers = list(source.tickers) if source.tickers is not None else list(base.tickers)
    tickers = _union_portfolio_tickers(base_tickers, candidate_tickers)
    rw = research.research_window
    module_name = str(research.eval_bias_spec["module_name"])
    inclusion_root = (
        Path(research.output_root)
        / research.feature_type.value
        / module_name
        / research.portfolio_inclusion.output_subdir
    )
    dirs_policy = (
        EnsembleDirsPolicy.ALLOW_EMPTY
        if not ensemble_dirs
        else EnsembleDirsPolicy.REQUIRE_NON_EMPTY
    )
    cfg = replace(
        base,
        tickers=tickers,
        ensemble_dirs=ensemble_dirs,
        ensemble_dirs_policy=dirs_policy,
        weight_layer_method=source.weight_layer_method,
        weight_layer_kwargs=dict(source.weight_layer_kwargs),
        max_position_pct=source.max_position_pct,
        baseline_mode=source.baseline_mode,
        benchmark_ticker=resolve_portfolio_benchmark_ticker(
            baseline_mode=source.baseline_mode,
            portfolio_tickers=tickers,
            benchmark_ticker=source.benchmark_ticker,
        ),
        target_volatility=source.target_volatility,
        train_window=ResearchWindow(start=rw.train_start, end=rw.train_end),
        validation_window=ResearchWindow(start=rw.val_start, end=rw.val_end),
        output_root=inclusion_root,
        export_per_timeframe_tearsheets=source.export_per_timeframe_tearsheets,
        export_per_ensemble_tearsheets=source.export_per_ensemble_tearsheets,
        strict_cache_preflight=source.strict_cache_preflight,
    )
    if not ensemble_dirs:
        return cfg
    scoped = config_with_ensemble_dirs(cfg, ensemble_dirs)
    if not candidate_tickers:
        return scoped
    expanded = _union_portfolio_tickers(scoped.tickers, candidate_tickers)
    if expanded == scoped.tickers:
        return scoped
    return replace(
        scoped,
        tickers=expanded,
        benchmark_ticker=resolve_portfolio_benchmark_ticker(
            baseline_mode=scoped.baseline_mode,
            portfolio_tickers=expanded,
            benchmark_ticker=scoped.benchmark_ticker,
        ),
    )


def _dataclass_from_dict(cls: type[_DataclassT], payload: Mapping[str, object]) -> _DataclassT:
    field_names = {field.name for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
    filtered = {key: value for key, value in payload.items() if key in field_names}
    if cls is BootstrapCI:
        return cls(**filtered)  # type: ignore[call-arg, return-value]
    if cls is EmpiricalComparisonResult and "delta_sr_ci" in filtered:
        ci_payload = filtered["delta_sr_ci"]
        if isinstance(ci_payload, Mapping):
            filtered["delta_sr_ci"] = _dataclass_from_dict(BootstrapCI, ci_payload)
        elif not isinstance(ci_payload, BootstrapCI):
            raise TypeError("empirical_comparison.delta_sr_ci must be a mapping or BootstrapCI")
    return cls(**filtered)  # type: ignore[call-arg, return-value]


def load_portfolio_addition_report_from_json(path: Path) -> PortfolioAdditionReport | None:
    """Rehydrate a persisted portfolio addition gate report."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("skipped"):
        return None
    if "risk_impact" not in payload:
        return None
    return PortfolioAdditionReport(
        pairwise_redundancy=_dataclass_from_dict(
            PairwiseRedundancyResult,
            payload["pairwise_redundancy"],
        ),
        analytical_hurdle=_dataclass_from_dict(
            AnalyticalHurdleResult,
            payload["analytical_hurdle"],
        ),
        empirical_comparison=_dataclass_from_dict(
            EmpiricalComparisonResult,
            payload["empirical_comparison"],
        ),
        risk_impact=_dataclass_from_dict(
            PortfolioRiskImpactResult,
            payload["risk_impact"],
        ),
        weight_assessment=_dataclass_from_dict(
            WeightAssessmentResult,
            payload["weight_assessment"],
        ),
        idm_improvement=_dataclass_from_dict(
            IDMImprovementResult,
            payload["idm_improvement"],
        ),
        passed=bool(payload["passed"]),
        weight_warning=bool(payload["weight_warning"]),
        interpretation=str(payload["interpretation"]),
        schema_version=str(payload.get("schema_version", "1.0")),
    )


def _train_val_concat(phase_train: PhaseResult, phase_val: PhaseResult) -> pd.Series:
    return _concat_returns(
        phase_train.combined_strategy_returns,
        phase_val.combined_strategy_returns,
    )


def _neutral_first_in_sleeve_risk_impact(
    *,
    max_dd_tolerance: float,
    max_ulcer_increase: float,
    stress_max_dd_tolerance: float,
) -> PortfolioRiskImpactResult:
    """Placeholder risk block when the sleeve has no prior return stream (SR-only gate)."""
    return PortfolioRiskImpactResult(
        max_dd_without=0.0,
        max_dd_with=0.0,
        delta_max_dd=0.0,
        ulcer_without=0.0,
        ulcer_with=0.0,
        delta_ulcer=0.0,
        sortino_without=0.0,
        sortino_with=0.0,
        delta_sortino=0.0,
        calmar_without=0.0,
        calmar_with=0.0,
        delta_calmar=0.0,
        stress_max_dd_without=0.0,
        stress_max_dd_with=0.0,
        delta_stress_max_dd=0.0,
        n_stress_periods=0,
        stress_metrics_reliable=False,
        passed_max_dd=True,
        passed_ulcer=True,
        passed_stress_max_dd=True,
        passed=True,
        max_dd_tolerance=float(max_dd_tolerance),
        max_ulcer_increase=float(max_ulcer_increase),
        stress_max_dd_tolerance=float(stress_max_dd_tolerance),
    )


def _is_flat_return_series(returns: np.ndarray) -> bool:
    if len(returns) == 0:
        return True
    return bool(np.allclose(returns, 0.0, atol=1e-15))


@dataclass(frozen=True)
class _GatePhaseTask:
    key: str
    portfolio_config: PortfolioResearchConfig
    phase: Literal["train", "validation"]


def _run_gate_phase_task(task: _GatePhaseTask) -> tuple[str, PhaseResult]:
    return task.key, run_single_phase_for_prop_firm(
        task.portfolio_config,
        task.phase,
        emit_tearsheets=False,
        run_preflight=False,
        run_purpose="metrics_only",
    )


def _run_gate_phase_tasks(
    tasks: Sequence[_GatePhaseTask],
    *,
    n_jobs: int,
) -> dict[str, PhaseResult]:
    if not tasks:
        return {}
    if n_jobs <= 1 or len(tasks) <= 1:
        return dict(_run_gate_phase_task(task) for task in tasks)
    from joblib import Parallel, delayed

    pairs = Parallel(n_jobs=min(n_jobs, len(tasks)), backend="loky")(
        delayed(_run_gate_phase_task)(task) for task in tasks
    )
    return dict(pairs)


def _flat_sleeve_phase_result(template: PhaseResult, *, name: str) -> PhaseResult:
    """Synthetic sleeve-without phase (zero strategy returns) for first-in-sleeve tearsheets."""
    zero_returns = pd.Series(0.0, index=template.combined_strategy_returns.index, dtype=float)
    empty_wl = pd.DataFrame(columns=list(template.weight_layer_diagnostics_df.columns))
    return replace(
        template,
        name=name,
        combined_strategy_returns=zero_returns,
        weight_layer_diagnostics_df=empty_wl,
    )


def _run_sleeve_portfolio_phases(
    portfolio_config: PortfolioResearchConfig,
    *,
    sleeve_baseline_dirs: Mapping[str, str],
    candidate_key: str,
    candidate_ensemble_dir: str,
    n_jobs: int = 1,
) -> tuple[
    pd.Series,
    pd.Series,
    PhaseResult | None,
    PhaseResult | None,
    PhaseResult,
    PhaseResult,
]:
    """Run sleeve-scoped portfolio phases (with = peers + candidate in the sleeve)."""
    sleeve_with_dirs = {**dict(sleeve_baseline_dirs), candidate_key: candidate_ensemble_dir}
    cfg_sleeve_with = config_with_ensemble_dirs(portfolio_config, sleeve_with_dirs)
    sleeve_tasks = [
        _GatePhaseTask("sleeve_with_tr", cfg_sleeve_with, "train"),
        _GatePhaseTask("sleeve_with_val", cfg_sleeve_with, "validation"),
    ]
    if sleeve_baseline_dirs:
        cfg_sleeve_base = config_with_ensemble_dirs(portfolio_config, sleeve_baseline_dirs)
        sleeve_tasks.extend(
            [
                _GatePhaseTask("sleeve_without_tr", cfg_sleeve_base, "train"),
                _GatePhaseTask("sleeve_without_val", cfg_sleeve_base, "validation"),
            ]
        )
    sleeve_results = _run_gate_phase_tasks(sleeve_tasks, n_jobs=n_jobs)
    sleeve_with_tr = sleeve_results["sleeve_with_tr"]
    sleeve_with_val = sleeve_results["sleeve_with_val"]
    sleeve_returns_with = _train_val_concat(sleeve_with_tr, sleeve_with_val)

    if sleeve_baseline_dirs:
        sleeve_without_tr = sleeve_results["sleeve_without_tr"]
        sleeve_without_val = sleeve_results["sleeve_without_val"]
        sleeve_returns_without = _train_val_concat(sleeve_without_tr, sleeve_without_val)
        return (
            sleeve_returns_without,
            sleeve_returns_with,
            sleeve_without_tr,
            sleeve_without_val,
            sleeve_with_tr,
            sleeve_with_val,
        )

    sleeve_returns_without = pd.Series(0.0, index=sleeve_returns_with.index, dtype=float)
    return (
        sleeve_returns_without,
        sleeve_returns_with,
        _flat_sleeve_phase_result(sleeve_with_tr, name="Sleeve train (empty)"),
        _flat_sleeve_phase_result(sleeve_with_val, name="Sleeve validation (empty)"),
        sleeve_with_tr,
        sleeve_with_val,
    )


def _normalize_gate_weight_assigned(weight: float) -> float:
    """Clamp a portfolio/sleeve weight fraction to the Core gate contract [0, 1]."""
    if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
        return 0.0
    return min(max(float(weight), 0.0), 1.0)


def _candidate_weight_assigned(
    with_train: PhaseResult,
    without_train: PhaseResult,
) -> float:
    with_df = with_train.weight_layer_diagnostics_df
    without_df = without_train.weight_layer_diagnostics_df
    if with_df.empty:
        return 0.0
    global_with = with_df[with_df["weight_layer_ticker"] == _GLOBAL_SYNTHETIC_TICKER]
    if global_with.empty:
        return 0.0
    without_ids = (
        set(without_df["stream_or_model_id"].astype(str))
        if not without_df.empty
        else set()
    )
    new_rows = global_with[~global_with["stream_or_model_id"].astype(str).isin(without_ids)]
    if new_rows.empty:
        return 0.0
    return _normalize_gate_weight_assigned(float(new_rows["stream_weight"].clip(lower=0).sum()))


def _build_pairwise_correlation_matrix(
    member_returns: Mapping[str, np.ndarray],
    *,
    candidate_key: str,
) -> pd.DataFrame:
    labels = [candidate_key, *[name for name in sorted(member_returns) if name != candidate_key]]
    size = len(labels)
    matrix = np.eye(size, dtype=float)
    for row_index, row_label in enumerate(labels):
        for col_index in range(row_index + 1, size):
            col_label = labels[col_index]
            corr = float(
                np.corrcoef(member_returns[row_label], member_returns[col_label])[0, 1]
            )
            matrix[row_index, col_index] = corr
            matrix[col_index, row_index] = corr
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _global_stream_weights(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "stream_or_model_id",
                "stream_source_model_name",
                "stream_weight",
                "weight_method",
                "cluster_id",
            ]
        )
    global_rows = df[df["weight_layer_ticker"] == _GLOBAL_SYNTHETIC_TICKER].copy()
    columns = [
        "stream_or_model_id",
        "stream_source_model_name",
        "stream_weight",
        "weight_method",
    ]
    if "cluster_id" in global_rows.columns:
        columns.append("cluster_id")
    return global_rows[columns]


def _hierarchy_segments_from_cluster_id(cluster_id: object) -> tuple[str | None, str | None, str | None]:
    """Return ``(asset_class, style_group, hierarchy_path)`` from a fitted assignment path."""
    if cluster_id is None or (isinstance(cluster_id, float) and pd.isna(cluster_id)):
        return None, None, None
    path = str(cluster_id).strip()
    if not path:
        return None, None, None
    parts = path.split("/")
    if parts and parts[0] == "root":
        parts = parts[1:]
    if len(parts) < 2:
        return None, None, path
    return parts[0], parts[1], path


def _sleeve_weight_assigned_fraction(
    weight_layer_with_df: pd.DataFrame,
    *,
    asset_class: str,
    style_group: str,
    candidate_stream_ids: frozenset[str],
) -> float:
    """Fraction of in-sleeve stream weight on candidate streams after refit."""
    streams = _global_stream_weights(weight_layer_with_df)
    if streams.empty:
        return 0.0
    sleeve_total = 0.0
    candidate_total = 0.0
    for _, row in streams.iterrows():
        sid = str(row["stream_or_model_id"])
        asset, style, _ = _hierarchy_segments_from_cluster_id(row.get("cluster_id"))
        if asset is None or style is None:
            asset = _asset_class_for_stream(sid, style_group)
            style = style_group
        if asset != asset_class or style != style_group:
            continue
        weight = max(0.0, float(row["stream_weight"]))
        sleeve_total += weight
        if sid in candidate_stream_ids:
            candidate_total += weight
    if sleeve_total <= 0.0:
        return 0.0
    return _normalize_gate_weight_assigned(candidate_total / sleeve_total)


def _resolve_sleeve_weight_assigned(
    *,
    sleeve_phase_with_tr: PhaseResult,
    sleeve_baseline_dirs: Mapping[str, str],
    candidate_key: str,
    candidate_ensemble_dir: str,
    candidate_stream_ids: frozenset[str],
    ticker_names: frozenset[str],
    sleeve_asset: str,
    sleeve_style: str,
) -> float:
    """Candidate share of sleeve weight using the sleeve-scoped refit diagnostics."""
    if not sleeve_baseline_dirs:
        return 1.0
    sleeve_merged = {**dict(sleeve_baseline_dirs), candidate_key: candidate_ensemble_dir}
    sleeve_scoped_ids = candidate_stream_ids_for_ensemble(
        _REPO_ROOT,
        candidate_ensemble_dir,
        portfolio_ticker_names=ticker_names,
        portfolio_ensemble_dirs=sleeve_merged,
    )
    stream_ids = sleeve_scoped_ids if sleeve_scoped_ids else candidate_stream_ids
    return _sleeve_weight_assigned_fraction(
        sleeve_phase_with_tr.weight_layer_diagnostics_df,
        asset_class=sleeve_asset,
        style_group=sleeve_style,
        candidate_stream_ids=stream_ids,
    )


def _asset_class_budgets_from_with_df(with_df: pd.DataFrame) -> dict[str, float]:
    streams = _global_stream_weights(with_df)
    if streams.empty or "cluster_id" not in streams.columns:
        return {}
    budgets: dict[str, float] = {}
    for _, row in streams.iterrows():
        asset, _, _ = _hierarchy_segments_from_cluster_id(row.get("cluster_id"))
        if asset is None:
            continue
        budgets[asset] = budgets.get(asset, 0.0) + float(row["stream_weight"])
    return budgets


def _style_group_budgets(streams: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Sum global stream weights by (asset_class, style_group)."""
    budgets: dict[tuple[str, str], float] = {}
    for _, row in streams.iterrows():
        asset, style, _ = _hierarchy_segments_from_cluster_id(row.get("cluster_id"))
        if asset is None or style is None:
            continue
        key = (asset, style)
        budgets[key] = budgets.get(key, 0.0) + float(row["stream_weight"])
    return budgets


def _style_group_stream_ids(streams: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], list[str]] = {}
    for stream_id, row in streams.set_index("stream_or_model_id").iterrows():
        asset, style, _ = _hierarchy_segments_from_cluster_id(row.get("cluster_id"))
        if asset is None or style is None:
            continue
        groups.setdefault((asset, style), []).append(str(stream_id))
    return {key: sorted(stream_ids) for key, stream_ids in groups.items()}


def _equal_split_display_weights(
    without_streams: pd.DataFrame,
    with_streams: pd.DataFrame,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Show equal share of each style-group budget (portfolio-addition UI semantics).

    Raw fitted weights can go negative under SR / inverse-correlation adjustment; the gate UI
    should show structural dilution from adding streams to a group, not signed tilt artifacts.
    """
    budget_without = _style_group_budgets(without_streams)
    budget_with = _style_group_budgets(with_streams)
    ids_without = _style_group_stream_ids(without_streams)
    ids_with = _style_group_stream_ids(with_streams)

    updated: list[dict[str, object]] = []
    for row in rows:
        asset = row.get("asset_class")
        style = row.get("style_group")
        if not isinstance(asset, str) or not isinstance(style, str) or not asset or not style:
            updated.append(row)
            continue
        key = (asset, style)
        stream_id = str(row.get("stream_id") or "")
        group_without = ids_without.get(key, [])
        group_with = ids_with.get(key, [])
        n_before = len(group_without)
        n_after = len(group_with)
        before = (
            budget_without.get(key, 0.0) / n_before
            if stream_id in group_without and n_before > 0
            else 0.0
        )
        after = (
            budget_with.get(key, 0.0) / n_after
            if stream_id in group_with and n_after > 0
            else 0.0
        )
        updated.append(
            {
                **row,
                "weight_without": before,
                "weight_with": after,
                "weight_delta": after - before,
            }
        )
    return updated


def _hierarchy_backed_weight_display(
    without_streams: pd.DataFrame,
    with_streams: pd.DataFrame,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Use equal-split semantics when both phases carry hierarchy cluster paths."""
    if (
        "cluster_id" in without_streams.columns
        and "cluster_id" in with_streams.columns
        and _style_group_stream_ids(without_streams)
    ):
        return _equal_split_display_weights(without_streams, with_streams, rows)
    return [
        {
            **row,
            "weight_without": max(0.0, float(row.get("weight_without") or 0.0)),
            "weight_with": max(0.0, float(row.get("weight_with") or 0.0)),
            "weight_delta": max(0.0, float(row.get("weight_with") or 0.0))
            - max(0.0, float(row.get("weight_without") or 0.0)),
        }
        for row in rows
    ]


def build_weight_layer_member_table(
    without_df: pd.DataFrame,
    with_df: pd.DataFrame,
    *,
    candidate_key: str,
) -> pd.DataFrame:
    """Long-format stream weights before/after refit for UI and CSV export."""

    without_streams = _global_stream_weights(without_df).set_index("stream_or_model_id")
    with_streams = _global_stream_weights(with_df).set_index("stream_or_model_id")
    stream_ids = sorted(set(without_streams.index.astype(str)) | set(with_streams.index.astype(str)))
    rows: list[dict[str, object]] = []
    for stream_id in stream_ids:
        before = (
            float(without_streams.loc[stream_id, "stream_weight"])
            if stream_id in without_streams.index
            else 0.0
        )
        after = (
            float(with_streams.loc[stream_id, "stream_weight"])
            if stream_id in with_streams.index
            else 0.0
        )
        model_name = (
            str(with_streams.loc[stream_id, "stream_source_model_name"])
            if stream_id in with_streams.index
            else str(without_streams.loc[stream_id, "stream_source_model_name"])
        )
        cluster_raw = (
            with_streams.loc[stream_id, "cluster_id"]
            if stream_id in with_streams.index and "cluster_id" in with_streams.columns
            else (
                without_streams.loc[stream_id, "cluster_id"]
                if stream_id in without_streams.index and "cluster_id" in without_streams.columns
                else None
            )
        )
        asset_class, style_group, hierarchy_path = _hierarchy_segments_from_cluster_id(cluster_raw)
        rows.append(
            {
                "stream_id": stream_id,
                "model_name": model_name,
                "hierarchy_path": hierarchy_path,
                "asset_class": asset_class,
                "style_group": style_group,
                "weight_without": before,
                "weight_with": after,
                "weight_delta": after - before,
                "is_new_stream": stream_id not in without_streams.index,
                "is_candidate_stream": stream_id not in without_streams.index,
                "candidate_key": candidate_key,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "stream_id",
                "model_name",
                "hierarchy_path",
                "asset_class",
                "style_group",
                "weight_without",
                "weight_with",
                "weight_delta",
                "is_new_stream",
                "is_candidate_stream",
                "candidate_key",
            ]
        )
    frame = pd.DataFrame(rows).sort_values(
        ["asset_class", "style_group", "weight_with", "weight_without"],
        ascending=[True, True, False, False],
        na_position="last",
    )
    display_rows = _hierarchy_backed_weight_display(
        without_streams.reset_index(),
        with_streams.reset_index(),
        frame.to_dict(orient="records"),
    )
    return pd.DataFrame(display_rows).sort_values(
        ["asset_class", "style_group", "weight_with", "weight_without"],
        ascending=[True, True, False, False],
        na_position="last",
    )


def build_pairwise_peer_rows(
    corr_matrix: pd.DataFrame,
    *,
    candidate_key: str,
    flag_threshold: float,
) -> list[dict[str, object]]:
    if corr_matrix.empty or candidate_key not in corr_matrix.index:
        return []
    return [
        {
            "peer_member": peer,
            "pearson_vs_candidate": float(corr_matrix.loc[candidate_key, peer]),
            "flagged": float(corr_matrix.loc[candidate_key, peer]) > flag_threshold,
        }
        for peer in corr_matrix.columns
        if peer != candidate_key
    ]


def _fdm_from_diagnostics(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    global_rows = df[df["weight_layer_ticker"] == _GLOBAL_SYNTHETIC_TICKER]
    if global_rows.empty or "fdm_multiplier" not in global_rows.columns:
        return None
    values = [
        float(v)
        for v in global_rows["fdm_multiplier"].dropna().unique()
        if isinstance(v, (int, float))
    ]
    return values[0] if values else None


def _parse_stream_ticker(stream_id: str) -> str | None:
    parts = str(stream_id).strip().split("::", 2)
    return parts[0].strip().upper() if parts and parts[0].strip() else None


def build_weight_layer_tree_from_members(
    member_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Nest stream rows into asset → style → stream for UI rendering."""
    assets: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in member_records:
        asset = str(row.get("asset_class") or "unknown")
        style = str(row.get("style_group") or "unknown")
        assets.setdefault(asset, {}).setdefault(style, []).append(
            {
                "type": "stream",
                "stream_id": row.get("stream_id"),
                "model_name": row.get("model_name"),
                "ticker": _parse_stream_ticker(str(row.get("stream_id") or "")),
                "hierarchy_path": row.get("hierarchy_path"),
                "weight_without": float(row.get("weight_without") or 0.0),
                "weight_with": float(row.get("weight_with") or 0.0),
                "weight_delta": float(row.get("weight_delta") or 0.0),
                "is_candidate_stream": bool(row.get("is_candidate_stream")),
                "is_new_stream": bool(row.get("is_new_stream")),
            }
        )

    tree: list[dict[str, object]] = []
    for asset_id in sorted(assets.keys()):
        style_map = assets[asset_id]
        style_nodes: list[dict[str, object]] = []
        for style_id in sorted(style_map.keys()):
            streams = sorted(
                style_map[style_id],
                key=lambda s: float(s.get("weight_with") or 0.0),
                reverse=True,
            )
            style_nodes.append(
                {
                    "type": "style",
                    "id": style_id,
                    "weight_without": sum(float(s["weight_without"]) for s in streams),
                    "weight_with": sum(float(s["weight_with"]) for s in streams),
                    "weight_delta": sum(float(s["weight_delta"]) for s in streams),
                    "stream_count": len(streams),
                    "children": streams,
                }
            )
        tree.append(
            {
                "type": "asset",
                "id": asset_id,
                "weight_without": sum(float(s["weight_without"]) for s in style_nodes),
                "weight_with": sum(float(s["weight_with"]) for s in style_nodes),
                "weight_delta": sum(float(s["weight_delta"]) for s in style_nodes),
                "stream_count": sum(int(s["stream_count"]) for s in style_nodes),
                "children": style_nodes,
            }
        )
    return tree


def _hierarchy_outline_from_spec(spec: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten hierarchy JSON into outline rows for the UI (path + leaf count)."""
    rows: list[dict[str, object]] = []

    def _walk(node: Mapping[str, object], path_parts: tuple[str, ...]) -> None:
        node_type = str(node.get("type", ""))
        if node_type == "leaf":
            sid = str(node.get("stream_id", "")).strip()
            rows.append(
                {
                    "path": "/".join(path_parts),
                    "kind": "leaf",
                    "stream_id": sid,
                    "ticker": _parse_stream_ticker(sid),
                }
            )
            return
        if node_type != "group":
            return
        group_id = str(node.get("id", "")).strip()
        next_path = path_parts + (group_id,) if group_id else path_parts
        children = node.get("children")
        if not isinstance(children, list):
            return
        leaf_count = sum(
            1
            for ch in children
            if isinstance(ch, Mapping) and str(ch.get("type")) == "leaf"
        )
        group_count = sum(
            1
            for ch in children
            if isinstance(ch, Mapping) and str(ch.get("type")) == "group"
        )
        rows.append(
            {
                "path": "/".join(next_path),
                "kind": "group",
                "group_id": group_id,
                "child_groups": group_count,
                "child_leaves": leaf_count,
            }
        )
        for child in children:
            if isinstance(child, Mapping):
                _walk(child, next_path)

    _walk(spec, ())
    return rows


def build_weight_layer_hierarchy_detail(
    inputs: PortfolioGateInputs,
    *,
    weight_layer_method: str,
    weight_layer_kwargs: Mapping[str, object],
    member_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Serializable weight-layer structure for portfolio-addition UI."""
    detail: dict[str, object] = {
        "method": weight_layer_method,
        "fdm_max": weight_layer_kwargs.get("fdm_max", 2.0),
        "fdm_without": _fdm_from_diagnostics(inputs.weight_layer_without_df),
        "fdm_with": _fdm_from_diagnostics(inputs.weight_layer_with_df),
        "candidate_key": inputs.candidate_key,
        "ensemble_count_without": len(inputs.baseline_ensemble_dirs),
        "ensemble_count_with": len(inputs.merged_ensemble_dirs),
        "stream_count_without": len(
            _global_stream_weights(inputs.weight_layer_without_df)
        ),
        "stream_count_with": len(_global_stream_weights(inputs.weight_layer_with_df)),
        "asset_budgets_without": _asset_class_budgets_from_with_df(
            inputs.weight_layer_without_df
        ),
        "asset_budgets_with": _asset_class_budgets_from_with_df(inputs.weight_layer_with_df),
        "tree": build_weight_layer_tree_from_members(member_records),
    }

    candidate_rows = [
        row for row in member_records if bool(row.get("is_candidate_stream"))
    ]
    if candidate_rows:
        detail["candidate_streams"] = [
            {
                "stream_id": row.get("stream_id"),
                "model_name": row.get("model_name"),
                "asset_class": row.get("asset_class"),
                "style_group": row.get("style_group"),
                "weight_with": float(row.get("weight_with") or 0.0),
                "hierarchy_path": row.get("hierarchy_path"),
            }
            for row in candidate_rows
        ]
        detail["candidate_weight_total"] = sum(
            float(row.get("weight_with") or 0.0) for row in candidate_rows
        )

    if weight_layer_method != "hierarchy_equal":
        return detail

    detail["hierarchy_mode"] = "asset_first"
    spec_without = inputs.hierarchy_spec_without
    spec_with = inputs.hierarchy_spec_with
    if spec_without is not None:
        detail["hierarchy_spec_without"] = spec_without
        detail["hierarchy_outline_without"] = _hierarchy_outline_from_spec(spec_without)
    if spec_with is not None:
        detail["hierarchy_spec_with"] = spec_with
        detail["hierarchy_outline_with"] = _hierarchy_outline_from_spec(spec_with)

    return detail


def build_drawdown_detail_payload(
    hurdle: AnalyticalHurdleResult,
) -> dict[str, object]:
    uplift = hurdle.corr_drawdown_conditional - hurdle.corr_unconditional
    return {
        "corr_unconditional": hurdle.corr_unconditional,
        "corr_drawdown_conditional": hurdle.corr_drawdown_conditional,
        "corr_effective": hurdle.corr_effective,
        "corr_drawdown_uplift": uplift,
        "drawdown_overlap": hurdle.drawdown_overlap,
        "joint_drawdown_depth": hurdle.joint_drawdown_depth,
        "drawdown_corr_uplift_flagged": hurdle.drawdown_corr_uplift_flagged,
        "drawdown_overlap_flagged": hurdle.drawdown_overlap_flagged,
        "joint_dd_depth_flagged": hurdle.joint_dd_depth_flagged,
        "drawdown_threshold": hurdle.drawdown_threshold,
        "drawdown_window": hurdle.drawdown_window,
        "n_stress_periods": hurdle.n_stress_periods,
    }


def build_portfolio_addition_context(
    inputs: PortfolioGateInputs,
    report: PortfolioAdditionReport,
    *,
    weight_layer_method: str,
    pairwise_flag_threshold: float,
    weight_layer_kwargs: Mapping[str, object] | None = None,
    portfolio_report: PortfolioAdditionReport | None = None,
) -> dict[str, object]:
    """Serializable context for UI: peers, drawdown detail, weight-layer competition."""

    peer_rows = build_pairwise_peer_rows(
        inputs.pairwise_corr_matrix,
        candidate_key=inputs.candidate_key,
        flag_threshold=pairwise_flag_threshold,
    )
    weight_members = build_weight_layer_member_table(
        inputs.weight_layer_without_df,
        inputs.weight_layer_with_df,
        candidate_key=inputs.candidate_key,
    )
    member_records = weight_members.to_dict(orient="records")
    wl_kw = dict(weight_layer_kwargs or {})
    context: dict[str, object] = {
        "candidate_key": inputs.candidate_key,
        "weight_layer_method": weight_layer_method,
        "pairwise_peers": peer_rows,
        "drawdown_detail": build_drawdown_detail_payload(report.analytical_hurdle),
        "weight_layer_members": member_records,
        "weight_layer_detail": build_weight_layer_hierarchy_detail(
            inputs,
            weight_layer_method=weight_layer_method,
            weight_layer_kwargs=wl_kw,
            member_records=member_records,
        ),
        "correlation_matrix_labels": (
            list(inputs.pairwise_corr_matrix.index.astype(str))
            if not inputs.pairwise_corr_matrix.empty
            else []
        ),
    }
    if weight_layer_method == "hierarchy_equal":
        context["weight_layer_hierarchy_mode"] = "asset_first"
        budgets = _asset_class_budgets_from_with_df(inputs.weight_layer_with_df)
        if budgets:
            context["asset_class_budgets"] = budgets
        budgets_before = _asset_class_budgets_from_with_df(inputs.weight_layer_without_df)
        if budgets_before:
            context["asset_class_budgets_without"] = budgets_before
    context["primary_gate_scope"] = "sleeve"
    context["sleeve_label"] = sleeve_label(inputs.sleeve_asset_class, inputs.sleeve_style_group)
    context["sleeve_peer_ensembles"] = sorted(inputs.sleeve_baseline_ensemble_dirs.keys())
    context["sleeve_weight_assigned"] = inputs.sleeve_weight_assigned
    if portfolio_report is not None:
        context["portfolio_gate"] = {
            "passed": portfolio_report.passed,
            "empirical_comparison": portfolio_report.empirical_comparison.to_json_dict(),
            "risk_impact": portfolio_report.risk_impact.to_json_dict(),
            "gate_criteria": portfolio_report.to_json_dict().get("gate_criteria"),
            "analytical_hurdle": portfolio_report.analytical_hurdle.to_json_dict(),
            "pairwise_redundancy": portfolio_report.pairwise_redundancy.to_json_dict(),
            "idm_improvement": portfolio_report.idm_improvement.to_json_dict(),
            "weight_assessment": portfolio_report.weight_assessment.to_json_dict(),
        }
    return context


def _risk_impact_scorecard_rows(
    report: PortfolioAdditionReport,
    *,
    scope: str,
) -> list[dict[str, object]]:
    risk = report.risk_impact
    empirical = report.empirical_comparison
    criteria = report.to_json_dict().get("gate_criteria", {})
    gate_criteria = criteria if isinstance(criteria, dict) else {}
    return [
        {
            "scope": scope,
            "metric": "sharpe",
            "without": empirical.sr_without,
            "with": empirical.sr_with,
            "delta": empirical.delta_sr,
            "passed": empirical.passed,
            "in_composite_gate": True,
        },
        {
            "scope": scope,
            "metric": "max_drawdown",
            "without": risk.max_dd_without,
            "with": risk.max_dd_with,
            "delta": risk.delta_max_dd,
            "passed": risk.passed_max_dd,
            "in_composite_gate": True,
        },
        {
            "scope": scope,
            "metric": "ulcer_index",
            "without": risk.ulcer_without,
            "with": risk.ulcer_with,
            "delta": risk.delta_ulcer,
            "passed": risk.passed_ulcer,
            "in_composite_gate": True,
        },
        {
            "scope": scope,
            "metric": "stress_max_drawdown",
            "without": risk.stress_max_dd_without,
            "with": risk.stress_max_dd_with,
            "delta": risk.delta_stress_max_dd,
            "passed": risk.passed_stress_max_dd,
            "in_composite_gate": risk.stress_metrics_reliable,
            "stress_metrics_reliable": risk.stress_metrics_reliable,
        },
        {
            "scope": scope,
            "metric": "sortino",
            "without": risk.sortino_without,
            "with": risk.sortino_with,
            "delta": risk.delta_sortino,
            "passed": None,
            "in_composite_gate": False,
        },
        {
            "scope": scope,
            "metric": "calmar",
            "without": risk.calmar_without,
            "with": risk.calmar_with,
            "delta": risk.delta_calmar,
            "passed": None,
            "in_composite_gate": False,
        },
        {
            "scope": scope,
            "metric": "composite_gate",
            "without": None,
            "with": None,
            "delta": None,
            "passed": report.passed,
            "in_composite_gate": True,
            "delta_sr_pass": gate_criteria.get("delta_sr"),
            "max_dd_pass": gate_criteria.get("max_dd"),
            "ulcer_pass": gate_criteria.get("ulcer"),
            "stress_max_dd_pass": gate_criteria.get("stress_max_dd"),
        },
    ]


def write_portfolio_addition_context_csvs(
    inputs: PortfolioGateInputs,
    report: PortfolioAdditionReport,
    output_dir: Path,
    *,
    portfolio_report: PortfolioAdditionReport | None = None,
) -> dict[str, Path]:
    """Write correlation, drawdown, and weight-layer CSV exports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if not inputs.pairwise_corr_matrix.empty:
        corr_path = output_dir / "portfolio_addition_pairwise_corr_matrix.csv"
        inputs.pairwise_corr_matrix.to_csv(corr_path)
        paths["pairwise_corr_matrix_csv"] = corr_path
        long_rows = [
            {
                "strategy_a": row_label,
                "strategy_b": col_label,
                "pearson": float(inputs.pairwise_corr_matrix.loc[row_label, col_label]),
            }
            for row_label in inputs.pairwise_corr_matrix.index
            for col_label in inputs.pairwise_corr_matrix.columns
            if row_label <= col_label
        ]
        long_path = output_dir / "portfolio_addition_pairwise_corr_long.csv"
        pd.DataFrame(long_rows).to_csv(long_path, index=False)
        paths["pairwise_corr_long_csv"] = long_path

    dd_path = output_dir / "portfolio_addition_drawdown_detail.csv"
    pd.DataFrame([build_drawdown_detail_payload(report.analytical_hurdle)]).to_csv(
        dd_path,
        index=False,
    )
    paths["drawdown_detail_csv"] = dd_path

    weight_path = output_dir / "portfolio_addition_weight_layer_members.csv"
    build_weight_layer_member_table(
        inputs.weight_layer_without_df,
        inputs.weight_layer_with_df,
        candidate_key=inputs.candidate_key,
    ).to_csv(weight_path, index=False)
    paths["weight_layer_members_csv"] = weight_path

    risk_rows = _risk_impact_scorecard_rows(report, scope="sleeve")
    if portfolio_report is not None:
        risk_rows.extend(_risk_impact_scorecard_rows(portfolio_report, scope="portfolio"))
    risk_path = output_dir / "portfolio_risk_impact.csv"
    pd.DataFrame(risk_rows).to_csv(risk_path, index=False)
    paths["portfolio_risk_impact_csv"] = risk_path
    return paths


def _align_series_to_arrays(series_map: Mapping[str, pd.Series]) -> dict[str, np.ndarray]:
    indexed = {key: series.sort_index() for key, series in series_map.items()}
    common_index = next(iter(indexed.values())).index
    for series in indexed.values():
        common_index = common_index.intersection(series.index)
    if len(common_index) == 0:
        raise ValueError("No overlapping dates across portfolio gate return series")
    return {
        key: np.asarray(series.loc[common_index].astype(float).values, dtype=float)
        for key, series in indexed.items()
    }


def collect_portfolio_gate_inputs(
    portfolio_config: PortfolioResearchConfig,
    *,
    candidate_ensemble_dir: str,
    candidate_key: str,
    candidate_weight_hierarchy_group: str,
    run_preflight: bool = True,
    n_jobs: int = 1,
) -> PortfolioGateInputs:
    """Run portfolio phases and assemble aligned arrays for the Core gate."""

    baseline_dirs = dict(portfolio_config.ensemble_dirs)
    if candidate_key in baseline_dirs:
        raise ValueError(f"Candidate key {candidate_key!r} already exists in baseline portfolio")

    merged_dirs = {**baseline_dirs, candidate_key: candidate_ensemble_dir}
    ticker_names = frozenset(
        t.name if hasattr(t, "name") else str(t) for t in portfolio_config.tickers
    )
    sleeve_ticker_names = candidate_ticker_names_from_ensemble(
        _REPO_ROOT,
        candidate_ensemble_dir,
        portfolio_ticker_names=ticker_names,
    )
    if not sleeve_ticker_names:
        sleeve_ticker_names = ticker_names
    sleeve_asset, sleeve_style = candidate_sleeve_identity(
        weight_hierarchy_group=candidate_weight_hierarchy_group,
        candidate_ticker_names=sleeve_ticker_names,
    )
    sleeve_baseline_dirs = ensemble_dirs_in_sleeve(
        _REPO_ROOT,
        baseline_dirs,
        asset_class=sleeve_asset,
        style_group=sleeve_style,
        portfolio_ticker_names=ticker_names,
    )

    phase_candidate_tr: PhaseResult | None = None
    phase_candidate_val: PhaseResult | None = None
    candidate_returns: pd.Series | None = None

    if not baseline_dirs:
        cfg_candidate = config_with_ensemble_dirs(
            portfolio_config,
            {candidate_key: candidate_ensemble_dir},
        )
        if run_preflight:
            run_portfolio_research_cache_preflight(cfg_candidate)
        candidate_results = _run_gate_phase_tasks(
            [
                _GatePhaseTask("candidate_tr", cfg_candidate, "train"),
                _GatePhaseTask("candidate_val", cfg_candidate, "validation"),
            ],
            n_jobs=n_jobs,
        )
        phase_candidate_tr = candidate_results["candidate_tr"]
        phase_candidate_val = candidate_results["candidate_val"]
        candidate_returns = _train_val_concat(phase_candidate_tr, phase_candidate_val)
        empty_wl = pd.DataFrame(
            columns=list(phase_candidate_tr.weight_layer_diagnostics_df.columns),
        )
        phase_without_tr = replace(
            phase_candidate_tr,
            name="train_without",
            weight_layer_diagnostics_df=empty_wl,
        )
        phase_without_val = replace(
            phase_candidate_val,
            name="validation_without",
            weight_layer_diagnostics_df=empty_wl,
        )
        phase_with_tr = phase_candidate_tr
        phase_with_val = phase_candidate_val
        portfolio_without = pd.Series(0.0, index=candidate_returns.index, dtype=float)
        portfolio_with = candidate_returns
        hierarchy_spec_without = None
        hierarchy_spec_with = (
            dict(cfg_candidate.weight_layer_kwargs["hierarchy_spec"])
            if cfg_candidate.weight_layer_method == "hierarchy_equal"
            and isinstance(cfg_candidate.weight_layer_kwargs.get("hierarchy_spec"), Mapping)
            else None
        )
        existing_returns: dict[str, pd.Series] = {}
    else:
        cfg_base = config_with_ensemble_dirs(portfolio_config, baseline_dirs)
        cfg_with = config_with_ensemble_dirs(portfolio_config, merged_dirs)
        hierarchy_spec_without = (
            dict(cfg_base.weight_layer_kwargs["hierarchy_spec"])
            if cfg_base.weight_layer_method == "hierarchy_equal"
            and isinstance(cfg_base.weight_layer_kwargs.get("hierarchy_spec"), Mapping)
            else None
        )
        hierarchy_spec_with = (
            dict(cfg_with.weight_layer_kwargs["hierarchy_spec"])
            if cfg_with.weight_layer_method == "hierarchy_equal"
            and isinstance(cfg_with.weight_layer_kwargs.get("hierarchy_spec"), Mapping)
            else None
        )
        if run_preflight:
            run_portfolio_research_cache_preflight(cfg_with)

        book_tasks = [
            _GatePhaseTask("without_tr", cfg_base, "train"),
            _GatePhaseTask("without_val", cfg_base, "validation"),
            _GatePhaseTask("with_tr", cfg_with, "train"),
            _GatePhaseTask("with_val", cfg_with, "validation"),
        ]
        book_tasks.extend(
            _GatePhaseTask(
                f"baseline::{name}::{phase}",
                config_with_ensemble_dirs(portfolio_config, {name: path}),
                phase,
            )
            for name, path in sorted(baseline_dirs.items())
            for phase in ("train", "validation")
        )
        book_results = _run_gate_phase_tasks(book_tasks, n_jobs=n_jobs)
        phase_without_tr = book_results["without_tr"]
        phase_without_val = book_results["without_val"]
        phase_with_tr = book_results["with_tr"]
        phase_with_val = book_results["with_val"]

        portfolio_without = _train_val_concat(phase_without_tr, phase_without_val)
        portfolio_with = _train_val_concat(phase_with_tr, phase_with_val)

        existing_returns = {
            name: _train_val_concat(
                book_results[f"baseline::{name}::train"],
                book_results[f"baseline::{name}::validation"],
            )
            for name in sorted(baseline_dirs)
        }

    candidate_streams = candidate_stream_ids_for_ensemble(
        _REPO_ROOT,
        candidate_ensemble_dir,
        portfolio_ticker_names=ticker_names,
        portfolio_ensemble_dirs=merged_dirs,
    )

    if not baseline_dirs:
        assert phase_candidate_tr is not None
        assert phase_candidate_val is not None
        assert candidate_returns is not None
        sleeve_returns_without = pd.Series(0.0, index=candidate_returns.index, dtype=float)
        sleeve_returns_with = candidate_returns
        sleeve_phase_without_tr = _flat_sleeve_phase_result(
            phase_candidate_tr,
            name="Sleeve train (empty)",
        )
        sleeve_phase_without_val = _flat_sleeve_phase_result(
            phase_candidate_val,
            name="Sleeve validation (empty)",
        )
        sleeve_phase_with_tr = phase_candidate_tr
        sleeve_phase_with_val = phase_candidate_val
        sleeve_weight = 1.0
    else:
        (
            sleeve_returns_without,
            sleeve_returns_with,
            sleeve_phase_without_tr,
            sleeve_phase_without_val,
            sleeve_phase_with_tr,
            sleeve_phase_with_val,
        ) = _run_sleeve_portfolio_phases(
            portfolio_config,
            sleeve_baseline_dirs=sleeve_baseline_dirs,
            candidate_key=candidate_key,
            candidate_ensemble_dir=candidate_ensemble_dir,
            n_jobs=n_jobs,
        )
        sleeve_weight = _resolve_sleeve_weight_assigned(
            sleeve_phase_with_tr=sleeve_phase_with_tr,
            sleeve_baseline_dirs=sleeve_baseline_dirs,
            candidate_key=candidate_key,
            candidate_ensemble_dir=candidate_ensemble_dir,
            candidate_stream_ids=candidate_streams,
            ticker_names=ticker_names,
            sleeve_asset=sleeve_asset,
            sleeve_style=sleeve_style,
        )

    if candidate_returns is None:
        if sleeve_baseline_dirs:
            cfg_candidate = config_with_ensemble_dirs(
                portfolio_config,
                {candidate_key: candidate_ensemble_dir},
            )
            if run_preflight:
                run_portfolio_research_cache_preflight(cfg_candidate)
            candidate_results = _run_gate_phase_tasks(
                [
                    _GatePhaseTask("candidate_tr", cfg_candidate, "train"),
                    _GatePhaseTask("candidate_val", cfg_candidate, "validation"),
                ],
                n_jobs=n_jobs,
            )
            phase_candidate_tr = candidate_results["candidate_tr"]
            phase_candidate_val = candidate_results["candidate_val"]
            candidate_returns = _train_val_concat(phase_candidate_tr, phase_candidate_val)
        else:
            phase_candidate_tr = sleeve_phase_with_tr
            phase_candidate_val = sleeve_phase_with_val
            candidate_returns = sleeve_returns_with

    series_map = {
        "new": candidate_returns,
        "portfolio_without": portfolio_without,
        "portfolio_with": portfolio_with,
        "sleeve_without": sleeve_returns_without,
        "sleeve_with": sleeve_returns_with,
        **{f"existing::{name}": series for name, series in existing_returns.items()},
    }
    aligned = _align_series_to_arrays(series_map)
    existing_strategy_returns = {
        name.split("::", 1)[1]: aligned[name]
        for name in aligned
        if name.startswith("existing::")
    }
    sleeve_existing = {
        name: existing_strategy_returns[name]
        for name in sleeve_baseline_dirs
        if name in existing_strategy_returns
    }

    member_returns = {candidate_key: aligned["new"], **existing_strategy_returns}
    pairwise_corr_matrix = _build_pairwise_correlation_matrix(
        member_returns,
        candidate_key=candidate_key,
    )

    eval_start = min(portfolio_without.index.min(), portfolio_with.index.min())
    eval_end = max(portfolio_without.index.max(), portfolio_with.index.max())

    return PortfolioGateInputs(
        new_strategy_returns=aligned["new"],
        existing_strategy_returns=existing_strategy_returns,
        portfolio_returns_without=aligned["portfolio_without"],
        portfolio_returns_with=aligned["portfolio_with"],
        weight_assigned=_candidate_weight_assigned(phase_with_tr, phase_without_tr),
        candidate_key=candidate_key,
        eval_start=pd.Timestamp(eval_start),
        eval_end=pd.Timestamp(eval_end),
        weight_layer_without_df=phase_without_tr.weight_layer_diagnostics_df.copy(),
        weight_layer_with_df=phase_with_tr.weight_layer_diagnostics_df.copy(),
        pairwise_corr_matrix=pairwise_corr_matrix,
        baseline_ensemble_dirs=dict(baseline_dirs),
        merged_ensemble_dirs=dict(merged_dirs),
        portfolio_ticker_names=ticker_names,
        sleeve_asset_class=sleeve_asset,
        sleeve_style_group=sleeve_style,
        sleeve_baseline_ensemble_dirs=dict(sleeve_baseline_dirs),
        sleeve_returns_without=aligned["sleeve_without"],
        sleeve_returns_with=aligned["sleeve_with"],
        sleeve_existing_strategy_returns=dict(sleeve_existing),
        sleeve_weight_assigned=sleeve_weight,
        candidate_stream_ids=candidate_streams,
        hierarchy_spec_without=hierarchy_spec_without,
        hierarchy_spec_with=hierarchy_spec_with,
        sleeve_phase_without_tr=sleeve_phase_without_tr,
        sleeve_phase_without_val=sleeve_phase_without_val,
        sleeve_phase_with_tr=sleeve_phase_with_tr,
        sleeve_phase_with_val=sleeve_phase_with_val,
        portfolio_phase_without_tr=phase_without_tr,
        portfolio_phase_without_val=phase_without_val,
        portfolio_phase_with_tr=phase_with_tr,
        portfolio_phase_with_val=phase_with_val,
        phase_candidate_tr=phase_candidate_tr,
        phase_candidate_val=phase_candidate_val,
    )


def _write_sleeve_tearsheet_artifacts(
    *,
    output_dir: Path,
    inputs: PortfolioGateInputs,
    gate_cfg: PortfolioAdditionGateConfig,
) -> tuple[Path, ...]:
    if not gate_cfg.emit_sleeve_tearsheets:
        return ()
    label = sleeve_label(inputs.sleeve_asset_class, inputs.sleeve_style_group)
    return write_sleeve_level_tearsheets(
        output_dir=output_dir,
        sleeve_label=label,
        phase_without_tr=inputs.sleeve_phase_without_tr,
        phase_without_val=inputs.sleeve_phase_without_val,
        phase_with_tr=inputs.sleeve_phase_with_tr,
        phase_with_val=inputs.sleeve_phase_with_val,
    )


def _write_full_portfolio_and_candidate_tearsheet_artifacts(
    *,
    output_dir: Path,
    inputs: PortfolioGateInputs,
    gate_cfg: PortfolioAdditionGateConfig,
) -> tuple[Path, ...]:
    if not gate_cfg.emit_tearsheets:
        return ()
    root = output_dir / "portfolio_gate_tearsheets"
    paths: list[Path] = []
    p_without_tr = inputs.portfolio_phase_without_tr
    p_without_val = inputs.portfolio_phase_without_val
    p_with_tr = inputs.portfolio_phase_with_tr
    p_with_val = inputs.portfolio_phase_with_val
    if (
        p_without_tr is not None
        and p_without_val is not None
        and p_with_tr is not None
        and p_with_val is not None
    ):
        skip_without = not _returns_suitable_for_quantstats_tearsheet(
            p_without_tr.combined_strategy_returns
        )
        paths.extend(
            write_full_portfolio_comparison_tearsheets(
                output_root=root,
                phase_without_tr=p_without_tr,
                phase_without_val=p_without_val,
                phase_with_tr=p_with_tr,
                phase_with_val=p_with_val,
                portfolio_ticker_names=inputs.portfolio_ticker_names,
                skip_without_tearsheets=skip_without,
            )
        )
    if inputs.phase_candidate_tr is not None and inputs.phase_candidate_val is not None:
        paths.extend(
            write_candidate_strategy_tearsheets(
                output_root=root,
                phase_candidate_tr=inputs.phase_candidate_tr,
                phase_candidate_val=inputs.phase_candidate_val,
                candidate_key=inputs.candidate_key,
            )
        )
    return tuple(paths)


def _write_portfolio_gate_tearsheet_artifacts(
    *,
    output_dir: Path,
    inputs: PortfolioGateInputs,
    gate_cfg: PortfolioAdditionGateConfig,
) -> tuple[Path, ...]:
    # Wipe any stale tearsheets from prior runs (path renames, retries, etc.)
    # before writing so the discovery never picks up old files alongside new ones.
    gate_root = output_dir / "portfolio_gate_tearsheets"
    if gate_root.exists():
        shutil.rmtree(gate_root)

    sleeve_paths = _write_sleeve_tearsheet_artifacts(
        output_dir=output_dir,
        inputs=inputs,
        gate_cfg=gate_cfg,
    )
    book_paths = _write_full_portfolio_and_candidate_tearsheet_artifacts(
        output_dir=output_dir,
        inputs=inputs,
        gate_cfg=gate_cfg,
    )
    return sleeve_paths + book_paths


def _annualised_sharpe(returns: np.ndarray, periods_per_year: int = _PERIODS_PER_YEAR) -> float:
    if len(returns) < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0
    return mean / std * math.sqrt(periods_per_year)


def block_bootstrap_delta_sr_samples(
    returns_without: np.ndarray,
    returns_with: np.ndarray,
    *,
    n_bootstrap: int,
    confidence: float = 0.95,
    periods_per_year: int = _PERIODS_PER_YEAR,
    seed: int | None = None,
) -> tuple[np.ndarray, int]:
    """Return bootstrap ΔSR samples using the same block scheme as Quant Foundry Core."""

    block_length = max(1, round(len(returns_without) ** (1.0 / 3.0)))
    n_blocks = len(returns_without) // block_length
    usable = n_blocks * block_length
    r_without = returns_without[:usable].reshape(n_blocks, block_length)
    r_with = returns_with[:usable].reshape(n_blocks, block_length)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        block_idx = rng.integers(0, n_blocks, size=n_blocks)
        samples[index] = (
            _annualised_sharpe(r_with[block_idx].ravel(), periods_per_year)
            - _annualised_sharpe(r_without[block_idx].ravel(), periods_per_year)
        )
    _ = confidence
    return samples, block_length


def _returns_are_degenerate(returns: np.ndarray) -> bool:
    if len(returns) < 2:
        return True
    return float(np.nanstd(returns, ddof=1)) < 1e-12


def _analytical_hurdle_for_sleeve(
    new_strategy_returns: np.ndarray,
    portfolio_returns_without: np.ndarray,
) -> AnalyticalHurdleResult:
    """Analytical hurdle; when the sleeve has no prior return stream, only SR_new applies."""
    if not _returns_are_degenerate(portfolio_returns_without):
        return compute_analytical_hurdle(new_strategy_returns, portfolio_returns_without)
    sr_new = _annualised_sharpe(new_strategy_returns)
    hurdle = 0.0
    return AnalyticalHurdleResult(
        sr_new=sr_new,
        sr_portfolio=0.0,
        corr_unconditional=0.0,
        corr_drawdown_conditional=0.0,
        corr_effective=0.0,
        hurdle=hurdle,
        margin=sr_new - hurdle,
        passed=sr_new > hurdle,
        drawdown_overlap=0.0,
        joint_drawdown_depth=0.0,
        drawdown_corr_uplift_flagged=False,
        drawdown_overlap_flagged=False,
        joint_dd_depth_flagged=False,
        drawdown_threshold=-0.05,
        drawdown_window=252,
        n_stress_periods=0,
    )


def _gate_kwargs_from_config(gate_cfg: PortfolioAdditionGateConfig) -> dict[str, object]:
    return {
        "pairwise_corr_flag_threshold": gate_cfg.pairwise_corr_flag_threshold,
        "delta_sr_threshold": gate_cfg.delta_sr_threshold,
        "max_dd_tolerance": gate_cfg.max_dd_tolerance_pp,
        "max_ulcer_increase": gate_cfg.max_ulcer_increase,
        "stress_max_dd_tolerance": gate_cfg.stress_max_dd_tolerance_pp,
        "weight_floor": gate_cfg.weight_floor,
        "n_bootstrap": gate_cfg.n_bootstrap,
        "bootstrap_seed": gate_cfg.random_seed,
    }


def _compute_first_in_sleeve_gate(
    new_strategy_returns: np.ndarray,
    portfolio_returns_without: np.ndarray,
    portfolio_returns_with: np.ndarray,
    *,
    weight_assigned: float,
    pairwise_corr_flag_threshold: float,
    delta_sr_threshold: float,
    max_dd_tolerance: float,
    max_ulcer_increase: float,
    stress_max_dd_tolerance: float,
    weight_floor: float,
    n_bootstrap: int,
    bootstrap_seed: int | None,
) -> PortfolioAdditionReport:
    """Gate when the candidate is the only strategy in its sleeve (no peer ensembles)."""
    pairwise = PairwiseRedundancyResult(
        max_pairwise_corr=0.0,
        most_similar_strategy="(first in sleeve)",
        flagged=False,
        corr_flag_threshold=float(pairwise_corr_flag_threshold),
        n_existing_strategies=1,
    )
    hurdle = _analytical_hurdle_for_sleeve(new_strategy_returns, portfolio_returns_without)
    empirical = compute_empirical_comparison(
        portfolio_returns_without,
        portfolio_returns_with,
        weight_assigned=weight_assigned,
        delta_sr_threshold=delta_sr_threshold,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    sleeve_without_is_empty = _is_flat_return_series(portfolio_returns_without)
    risk = (
        _neutral_first_in_sleeve_risk_impact(
            max_dd_tolerance=max_dd_tolerance,
            max_ulcer_increase=max_ulcer_increase,
            stress_max_dd_tolerance=stress_max_dd_tolerance,
        )
        if sleeve_without_is_empty
        else compute_portfolio_risk_impact(
            portfolio_returns_without,
            portfolio_returns_with,
            new_strategy_returns,
            max_dd_tolerance=max_dd_tolerance,
            max_ulcer_increase=max_ulcer_increase,
            stress_max_dd_tolerance=stress_max_dd_tolerance,
        )
    )
    weight = compute_weight_assessment(weight_assigned, weight_floor=weight_floor)
    existing_matrix = new_strategy_returns.reshape(-1, 1)
    idm = compute_idm_improvement(existing_matrix, new_strategy_returns)
    composite_passed = empirical.passed and risk.passed
    risk_detail = (
        "Risk vs empty sleeve baseline: SR gate only (drawdown/ulcer N/A). "
        if sleeve_without_is_empty
        else f"ΔmaxDD {risk.delta_max_dd:+.1%}, Δulcer {risk.delta_ulcer:+.3f}, "
    )
    interpretation = (
        f"First strategy in sleeve: ΔSR {empirical.delta_sr:+.3f} "
        f"(sleeve SR {empirical.sr_without:.2f} → {empirical.sr_with:.2f}), "
        f"{risk_detail}"
        f"weight {weight.weight_assigned:.1%}. "
        + (
            "Add to portfolio."
            if composite_passed
            else "Discard — composite gate failed (Sharpe and/or risk criteria)."
        )
    )
    return PortfolioAdditionReport(
        pairwise_redundancy=pairwise,
        analytical_hurdle=hurdle,
        empirical_comparison=empirical,
        risk_impact=risk,
        weight_assessment=weight,
        idm_improvement=idm,
        passed=composite_passed,
        weight_warning=not weight.meaningful,
        interpretation=interpretation,
    )


def compute_dual_scope_portfolio_addition_gate(
    inputs: PortfolioGateInputs,
    *,
    gate_cfg: PortfolioAdditionGateConfig,
) -> tuple[PortfolioAdditionReport, PortfolioAdditionReport]:
    """Run sleeve-scoped (primary) and full-portfolio (context) gate checks."""
    kwargs = _gate_kwargs_from_config(gate_cfg)

    sleeve_existing = inputs.sleeve_existing_strategy_returns
    if not sleeve_existing:
        sleeve_report = _compute_first_in_sleeve_gate(
            inputs.new_strategy_returns,
            inputs.sleeve_returns_without,
            inputs.sleeve_returns_with,
            weight_assigned=inputs.sleeve_weight_assigned,
            **kwargs,
        )
    else:
        sleeve_report = compute_portfolio_addition_gate(
            inputs.new_strategy_returns,
            sleeve_existing,
            inputs.sleeve_returns_without,
            inputs.sleeve_returns_with,
            weight_assigned=inputs.sleeve_weight_assigned,
            **kwargs,
        )

    if not inputs.existing_strategy_returns:
        portfolio_report = _compute_first_in_sleeve_gate(
            inputs.new_strategy_returns,
            inputs.portfolio_returns_without,
            inputs.portfolio_returns_with,
            weight_assigned=inputs.weight_assigned,
            **kwargs,
        )
    else:
        portfolio_report = compute_portfolio_addition_gate(
            inputs.new_strategy_returns,
            inputs.existing_strategy_returns,
            inputs.portfolio_returns_without,
            inputs.portfolio_returns_with,
            weight_assigned=inputs.weight_assigned,
            **kwargs,
        )

    label = sleeve_label(inputs.sleeve_asset_class, inputs.sleeve_style_group)
    interpretation = _build_dual_scope_interpretation(
        sleeve_report,
        portfolio_report,
        sleeve_label=label,
        n_sleeve_peers=len(inputs.sleeve_baseline_ensemble_dirs),
    )

    primary = PortfolioAdditionReport(
        pairwise_redundancy=sleeve_report.pairwise_redundancy,
        analytical_hurdle=sleeve_report.analytical_hurdle,
        empirical_comparison=sleeve_report.empirical_comparison,
        risk_impact=sleeve_report.risk_impact,
        weight_assessment=sleeve_report.weight_assessment,
        idm_improvement=sleeve_report.idm_improvement,
        passed=sleeve_report.passed,
        weight_warning=sleeve_report.weight_warning,
        interpretation=interpretation,
        schema_version=sleeve_report.schema_version,
    )
    return primary, portfolio_report


def dual_scope_gate_payload(
    report: PortfolioAdditionReport,
    inputs: PortfolioGateInputs,
    *,
    portfolio_report: PortfolioAdditionReport,
) -> dict[str, object]:
    """Flatten sleeve-primary report plus nested full-portfolio metrics for JSON."""
    base = report.to_json_dict()
    base["primary_gate_scope"] = "sleeve"
    base["sleeve_label"] = sleeve_label(inputs.sleeve_asset_class, inputs.sleeve_style_group)
    base["sleeve_asset_class"] = inputs.sleeve_asset_class
    base["sleeve_style_group"] = inputs.sleeve_style_group
    base["sleeve_peer_ensembles"] = sorted(inputs.sleeve_baseline_ensemble_dirs.keys())
    base["sleeve_weight_assigned"] = inputs.sleeve_weight_assigned
    base["portfolio_passed"] = portfolio_report.passed
    base["portfolio"] = portfolio_report.to_json_dict()
    return base


def _build_dual_scope_interpretation(
    sleeve_report: PortfolioAdditionReport,
    portfolio_report: PortfolioAdditionReport,
    *,
    sleeve_label: str,
    n_sleeve_peers: int,
) -> str:
    sleeve_emp = sleeve_report.empirical_comparison
    port_emp = portfolio_report.empirical_comparison
    peer_note = (
        f" ({n_sleeve_peers} existing ensemble(s) in this sleeve)"
        if n_sleeve_peers
        else " (first strategy in this sleeve)"
    )
    if sleeve_report.passed:
        if not portfolio_report.passed:
            return (
                f"Passes on sleeve {sleeve_label}{peer_note}: ΔSR "
                f"{sleeve_emp.delta_sr:+.3f} (sleeve SR {sleeve_emp.sr_without:.2f} → "
                f"{sleeve_emp.sr_with:.2f}). Full portfolio ΔSR "
                f"{port_emp.delta_sr:+.3f} is below the {port_emp.delta_sr_threshold:.2f} floor — "
                f"review cross-sleeve dilution before committing."
            )
        return (
            f"Passes on sleeve {sleeve_label}{peer_note}: ΔSR {sleeve_emp.delta_sr:+.3f}, "
            f"sleeve weight {sleeve_report.weight_assessment.weight_assigned:.1%}. "
            f"Full portfolio ΔSR {port_emp.delta_sr:+.3f} (SR {port_emp.sr_without:.2f} → "
            f"{port_emp.sr_with:.2f}). Add to portfolio."
        )
    if portfolio_report.passed:
        return (
            f"Fails sleeve {sleeve_label}{peer_note} (ΔSR {sleeve_emp.delta_sr:+.3f}, "
            f"required ≥ {sleeve_emp.delta_sr_threshold:.2f}) despite full-portfolio improvement "
            f"(ΔSR {port_emp.delta_sr:+.3f}). Discard — the candidate does not help its sleeve."
        )
    return (
        f"Fails on sleeve {sleeve_label}{peer_note}: ΔSR {sleeve_emp.delta_sr:+.3f}. "
        f"Full portfolio ΔSR {port_emp.delta_sr:+.3f}. Discard."
    )


def _gate_meta_payload(
    research: ResearchConfig,
    *,
    candidate_key: str,
    inputs: PortfolioGateInputs | None = None,
) -> dict[str, object]:
    rw = research.research_window
    if rw is None:
        raise ValueError("research_window is required")
    meta: dict[str, object] = {
        "candidate_key": candidate_key,
        "module_name": str(research.eval_bias_spec["module_name"]),
        "train_start": rw.train_start.date().isoformat(),
        "train_end": rw.train_end.date().isoformat(),
        "val_start": rw.val_start.date().isoformat(),
        "val_end": rw.val_end.date().isoformat(),
        "eval_start": (
            inputs.eval_start.date().isoformat()
            if inputs is not None
            else rw.train_start.date().isoformat()
        ),
        "eval_end": (
            inputs.eval_end.date().isoformat()
            if inputs is not None
            else rw.val_end.date().isoformat()
        ),
        "weight_layer_method": (
            research.portfolio_source.weight_layer_method
            if research.portfolio_source is not None
            else "ledoit_wolf_min_corr"
        ),
    }
    if inputs is not None:
        meta["n_obs"] = len(inputs.new_strategy_returns)
        meta["n_existing_strategies"] = len(inputs.existing_strategy_returns)
    return meta


def _first_strategy_skip_payload(research: ResearchConfig) -> dict[str, object]:
    inclusion = research.portfolio_inclusion
    candidate_key = (
        inclusion.candidate_key
        or inclusion.ephemeral_ensemble_name
    )
    return {
        "skipped": True,
        "reason": "first_strategy",
        "passed": True,
        "interpretation": (
            "This is the first strategy in this portfolio. Portfolio addition gate is skipped — "
            "there is no existing portfolio to compare against."
        ),
        "meta": _gate_meta_payload(research, candidate_key=candidate_key),
    }


def run_portfolio_addition_gate_pipeline(
    research: ResearchConfig,
) -> tuple[
    PortfolioAdditionReport | None,
    PortfolioAdditionReport | None,
    PortfolioGateInputs | None,
    str,
    Path | None,
]:
    """Run portfolio phases and compute the Core portfolio addition gate report."""

    if not research.portfolio_addition_gate.enabled:
        return None, None, None, "", None
    if research.research_window is None:
        raise ValueError("research_window is required for portfolio addition gate")

    portfolio_config = build_portfolio_config_for_gate(research)

    inclusion = research.portfolio_inclusion
    tmp_root: Path | None = None
    candidate_path: str
    candidate_key: str

    if inclusion.candidate_mode == "eval_bias_spec":
        candidate_key = inclusion.candidate_key or inclusion.ephemeral_ensemble_name
        candidate_path, tmp_root = materialize_inclusion_candidate_from_eval_bias_spec(
            research,
            portfolio_config,
            ephemeral_ensemble_name=inclusion.ephemeral_ensemble_name,
            weight_hierarchy_group=inclusion.ephemeral_weight_hierarchy_group,
        )
    else:
        if inclusion.candidate_repo_relative_path is None:
            raise ValueError(
                "portfolio_inclusion.candidate_repo_relative_path is required when "
                "candidate_mode='vault_path'"
            )
        candidate_path = inclusion.candidate_repo_relative_path
        candidate_key = inclusion.candidate_key or infer_candidate_key(candidate_path)

    try:
        inputs = collect_portfolio_gate_inputs(
            portfolio_config,
            candidate_ensemble_dir=candidate_path,
            candidate_key=candidate_key,
            candidate_weight_hierarchy_group=inclusion.ephemeral_weight_hierarchy_group,
            n_jobs=resolve_portfolio_gate_n_jobs(research),
            run_preflight=inclusion.preflight,
        )
        gate_cfg = research.portfolio_addition_gate
        primary_report, portfolio_report = compute_dual_scope_portfolio_addition_gate(
            inputs,
            gate_cfg=gate_cfg,
        )
        return primary_report, portfolio_report, inputs, candidate_key, tmp_root
    except Exception:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)
        raise


def _markdown_summary(report: PortfolioAdditionReport, *, candidate_key: str) -> str:
    empirical = report.empirical_comparison
    hurdle = report.analytical_hurdle
    pairwise = report.pairwise_redundancy
    ci = empirical.delta_sr_ci
    return "\n".join(
        [
            "# Portfolio addition gate",
            "",
            f"- Candidate: `{candidate_key}`",
            f"- Pairwise max corr: {pairwise.max_pairwise_corr:.4f} "
            f"({'flagged' if pairwise.flagged else 'ok'})",
            f"- Analytical hurdle margin: {hurdle.margin:+.4f} "
            f"({'pass' if hurdle.passed else 'fail'})",
            f"- Portfolio SR without: {empirical.sr_without:.4f}",
            f"- Portfolio SR with: {empirical.sr_with:.4f}",
            f"- ΔSR: {empirical.delta_sr:+.4f} "
            f"[{ci.lower:+.4f}, {ci.upper:+.4f}]",
            f"- Weight assigned: {empirical.weight_assigned:.4f}",
            f"- **Result:** {'PASS' if report.passed else 'FAIL'} — {report.interpretation}",
        ]
    )


def write_portfolio_addition_summary(
    payload: dict[str, object],
    output_dir: Path,
    *,
    bootstrap_samples: Sequence[float] | None = None,
) -> dict[str, Path]:
    """Persist portfolio addition gate JSON, markdown, and bootstrap CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "portfolio_addition_report.json"
    markdown_path = output_dir / "portfolio_addition_summary.md"
    bootstrap_csv = output_dir / "portfolio_addition_delta_sr_bootstrap.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload.get("skipped"):
        report = load_portfolio_addition_report_from_json(json_path)
        if report is not None:
            candidate_key = str(payload.get("meta", {}).get("candidate_key", "candidate"))
            markdown_path.write_text(
                _markdown_summary(report, candidate_key=candidate_key),
                encoding="utf-8",
            )
    else:
        markdown_path.write_text(str(payload.get("interpretation", "")), encoding="utf-8")

    if bootstrap_samples is not None:
        pd.DataFrame({"delta_sr": list(bootstrap_samples)}).to_csv(bootstrap_csv, index=False)
    elif bootstrap_csv.exists():
        bootstrap_csv.unlink()

    artifacts = {"json": json_path, "markdown": markdown_path}
    if bootstrap_samples is not None:
        artifacts["bootstrap_csv"] = bootstrap_csv
    return artifacts


def run_and_write_portfolio_addition_gate(
    research: ResearchConfig,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Run the portfolio addition gate and write artifacts under ``output_dir``."""

    if not research.portfolio_addition_gate.enabled:
        return {}

    primary_report, portfolio_report, inputs, candidate_key, ephemeral_tmp_root = (
        run_portfolio_addition_gate_pipeline(research)
    )
    if primary_report is None or portfolio_report is None or inputs is None:
        payload = _first_strategy_skip_payload(research)
        return write_portfolio_addition_summary(payload, output_dir)

    try:
        return _write_portfolio_addition_gate_artifacts(
            research,
            output_dir=output_dir,
            primary_report=primary_report,
            portfolio_report=portfolio_report,
            inputs=inputs,
            candidate_key=candidate_key,
        )
    finally:
        if ephemeral_tmp_root is not None:
            shutil.rmtree(ephemeral_tmp_root, ignore_errors=True)


def _write_portfolio_addition_gate_artifacts(
    research: ResearchConfig,
    *,
    output_dir: Path,
    primary_report: PortfolioAdditionReport,
    portfolio_report: PortfolioAdditionReport,
    inputs: PortfolioGateInputs,
    candidate_key: str,
) -> dict[str, Path]:
    gate_cfg = research.portfolio_addition_gate
    bootstrap_samples, _block_length = block_bootstrap_delta_sr_samples(
        inputs.portfolio_returns_without,
        inputs.portfolio_returns_with,
        n_bootstrap=gate_cfg.n_bootstrap,
        seed=gate_cfg.random_seed,
    )
    weight_layer_method = (
        research.portfolio_source.weight_layer_method
        if research.portfolio_source is not None
        else "ledoit_wolf_min_corr"
    )
    portfolio_config = build_portfolio_config_for_gate(research)
    wl_kwargs = (
        dict(research.portfolio_source.weight_layer_kwargs)
        if research.portfolio_source is not None
        else dict(portfolio_config.weight_layer_kwargs)
    )
    context = build_portfolio_addition_context(
        inputs,
        primary_report,
        weight_layer_method=weight_layer_method,
        pairwise_flag_threshold=gate_cfg.pairwise_corr_flag_threshold,
        weight_layer_kwargs=wl_kwargs,
        portfolio_report=portfolio_report,
    )
    gate_tearsheet_paths = _write_portfolio_gate_tearsheet_artifacts(
        output_dir=output_dir,
        inputs=inputs,
        gate_cfg=gate_cfg,
    )
    sleeve_tearsheet_paths = tuple(
        path
        for path in gate_tearsheet_paths
        if "/sleeve_" in path.as_posix() and "portfolio_gate_tearsheets" in path.as_posix()
    )
    portfolio_tearsheet_paths = tuple(
        path
        for path in gate_tearsheet_paths
        if "portfolio_gate_tearsheets" in path.as_posix()
    )
    sleeve_tearsheets_subdir: str | None = None
    portfolio_tearsheets_subdir: str | None = None
    if gate_tearsheet_paths:
        context = {**context}
        if sleeve_tearsheet_paths:
            sleeve_tearsheets_subdir = "/".join(
                sleeve_tearsheet_paths[0].relative_to(output_dir).parts[:2]
            )
            context["sleeve_tearsheet_files"] = [
                str(path.relative_to(output_dir)) for path in sleeve_tearsheet_paths
            ]
            context["sleeve_tearsheets_dir"] = sleeve_tearsheets_subdir
        if portfolio_tearsheet_paths:
            portfolio_tearsheets_subdir = portfolio_tearsheet_paths[0].relative_to(
                output_dir
            ).parts[0]
            context["portfolio_tearsheet_files"] = [
                str(path.relative_to(output_dir)) for path in portfolio_tearsheet_paths
            ]
            context["portfolio_tearsheets_dir"] = portfolio_tearsheets_subdir

    payload = {
        "skipped": False,
        "meta": _gate_meta_payload(research, candidate_key=candidate_key, inputs=inputs),
        "context": context,
        **dual_scope_gate_payload(
            primary_report,
            inputs,
            portfolio_report=portfolio_report,
        ),
    }
    artifacts = write_portfolio_addition_summary(
        payload,
        output_dir,
        bootstrap_samples=bootstrap_samples.tolist(),
    )
    artifacts.update(
        write_portfolio_addition_context_csvs(
            inputs,
            primary_report,
            output_dir,
            portfolio_report=portfolio_report,
        )
    )
    if sleeve_tearsheets_subdir is not None:
        artifacts["sleeve_tearsheets_dir"] = output_dir / sleeve_tearsheets_subdir
    if portfolio_tearsheets_subdir is not None:
        artifacts["portfolio_tearsheets_dir"] = output_dir / portfolio_tearsheets_subdir

    from research.feature.visualization.portfolio_addition_reports import (
        write_portfolio_addition_plots,
    )

    plot_dir = output_dir / "matplotlib"
    plot_paths = write_portfolio_addition_plots(
        payload,
        bootstrap_csv=artifacts.get("bootstrap_csv"),
        corr_matrix_csv=artifacts.get("pairwise_corr_matrix_csv"),
        drawdown_detail_csv=artifacts.get("drawdown_detail_csv"),
        output_dir=plot_dir,
    )
    artifacts["plot_dir"] = plot_dir
    for index, plot_path in enumerate(plot_paths):
        artifacts[f"plot_{index}"] = plot_path
    return artifacts


def refresh_portfolio_addition_plots(
    input_dir: Path,
    *,
    output_dir: Path | None = None,
) -> list[Path]:
    """Regenerate portfolio addition plots from persisted JSON/CSV artifacts."""

    report_path = input_dir / "portfolio_addition_report.json"
    bootstrap_csv = input_dir / "portfolio_addition_delta_sr_bootstrap.csv"
    if not report_path.exists():
        return []
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("skipped"):
        return []
    from research.feature.visualization.portfolio_addition_reports import (
        write_portfolio_addition_plots,
    )

    plot_dir = output_dir or (input_dir / "matplotlib")
    corr_csv = input_dir / "portfolio_addition_pairwise_corr_matrix.csv"
    dd_csv = input_dir / "portfolio_addition_drawdown_detail.csv"
    return write_portfolio_addition_plots(
        payload,
        bootstrap_csv=bootstrap_csv if bootstrap_csv.exists() else None,
        corr_matrix_csv=corr_csv if corr_csv.exists() else None,
        drawdown_detail_csv=dd_csv if dd_csv.exists() else None,
        output_dir=plot_dir,
    )
