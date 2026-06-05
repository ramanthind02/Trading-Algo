"""Min-step axis-aligned parameter perturbation for exploration."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from quantfoundry_core.robustness import (
    PerturbationTestResult,
    aggregate_perturbation_results,
    build_perturbation_neighbors,
)

from research.feature.binning.transforms import inflate_params_from_combo_long_table
from research.feature.config import FeatureType, ResearchConfig
from research.feature.in_sample.data_loader import (
    BIAS_MODULE_COMBO_KEY,
    first_bias_spec,
    load_features_for_combo,
    params_for_bias_node,
)
from research.feature.in_sample.metric_helpers import (
    SUPPORTED_PARAM_SENSITIVITY_METRICS,
    compute_param_sensitivity_metric,
)
from research.feature.research_table_exports import canonical_in_sample_visualization_dir


@dataclass(frozen=True)
class PerturbationRunRecord:
    role: str
    param_changed: str
    direction: str
    metric: float
    params: dict[str, object]


@dataclass(frozen=True)
class PerturbationPipelineResult:
    result: PerturbationTestResult
    runs: tuple[PerturbationRunRecord, ...]
    report_json: Path
    runs_csv: Path
    summary_md: Path


def exploration_perturbation_enabled(config: ResearchConfig) -> bool:
    """Whether min-step perturbation should run during exploration."""

    ps_cfg = config.param_sensitivity
    return (
        ps_cfg.perturbation_enabled
        and bool(ps_cfg.perturbation_specs)
        and config.feature_type == FeatureType.SIGNED_SIGNAL
    )


def _perturbation_param_view(
    params: Mapping[str, object],
    spec_keys: Iterable[str],
) -> dict[str, object]:
    """Map perturbation axis names to values (unwrap ``signal_params`` on filter gates)."""
    signal_params = params.get("signal_params")
    if isinstance(signal_params, Mapping):
        return {key: signal_params[key] for key in spec_keys if key in signal_params}
    return {key: params[key] for key in spec_keys if key in params}


def _merge_perturbation_param_view(
    params: Mapping[str, object],
    view: Mapping[str, object],
) -> dict[str, object]:
    """Write perturbed signal-leg values back into a full combo param dict."""
    merged = dict(params)
    signal_params = merged.get("signal_params")
    if isinstance(signal_params, Mapping):
        updated_signal = dict(signal_params)
        updated_signal.update(view)
        merged["signal_params"] = updated_signal
        return merged
    merged.update(view)
    return merged


def _combo_spec_for_params(config: ResearchConfig, params: Mapping[str, object]) -> dict[str, Any]:
    template = first_bias_spec(config.bias_spec)
    module_name = params.get(BIAS_MODULE_COMBO_KEY, template.get("module_name"))
    return {
        **template,
        "module_name": module_name,
        "params": params_for_bias_node(params),
    }


def _strategy_returns_for_params(
    config: ResearchConfig,
    params: Mapping[str, object],
) -> pd.Series | None:
    data = load_features_for_combo(
        _combo_spec_for_params(config, params),
        config,
        populate_on_miss=True,
    )
    if data is None:
        return None
    feature, target, _, _ = data
    paired = pd.concat(
        [feature.rename("signal"), target.rename("target")],
        axis=1,
    ).dropna(how="any")
    if paired.empty:
        return None
    return paired["signal"] * paired["target"]


def _metric_for_params(config: ResearchConfig, params: Mapping[str, object]) -> float:
    metric_name = config.param_sensitivity.perturbation_metric
    if metric_name not in SUPPORTED_PARAM_SENSITIVITY_METRICS:
        supported = ", ".join(sorted(SUPPORTED_PARAM_SENSITIVITY_METRICS))
        raise ValueError(
            f"Unsupported perturbation_metric {metric_name!r}. Supported: {supported}."
        )
    returns = _strategy_returns_for_params(config, params)
    if returns is None or returns.empty:
        return float("nan")
    return float(
        compute_param_sensitivity_metric(
            returns.to_numpy(dtype=float),
            metric_name,
            config.timeframe,
        )
    )


def _neighbor_delta(
    chosen: Mapping[str, object],
    neighbor: Mapping[str, object],
    *,
    spec_keys: Iterable[str],
) -> tuple[str, str]:
    chosen_view = _perturbation_param_view(chosen, spec_keys)
    neighbor_view = _perturbation_param_view(neighbor, spec_keys)
    for key in sorted(spec_keys):
        if neighbor_view.get(key) != chosen_view.get(key):
            direction = "up" if neighbor_view[key] > chosen_view[key] else "down"
            return key, direction
    return "", ""


def _coerce_chosen_params(params: Mapping[str, object]) -> dict[str, object]:
    return {str(key): value for key, value in params.items()}


def run_min_step_perturbation_pipeline(
    config: ResearchConfig,
    output_dir: Path,
    *,
    chosen_params: Mapping[str, object],
) -> PerturbationPipelineResult:
    """Evaluate axis-aligned min-step neighbours and write perturbation artifacts."""

    del output_dir  # artifacts use canonical visualization dir
    ps_cfg = config.param_sensitivity
    full_chosen = _coerce_chosen_params(
        inflate_params_from_combo_long_table(chosen_params)
    )
    chosen_view = _perturbation_param_view(
        full_chosen, ps_cfg.perturbation_specs
    )
    # Multi-module exploration catalogs may list axes for every branch (e.g. Donchian
    # ``lookback`` + Turtle ``entry_lookback``); only perturb keys on the winner.
    active_specs = {
        name: spec
        for name, spec in ps_cfg.perturbation_specs.items()
        if name in chosen_view
    }
    if not active_specs:
        raise ValueError(
            "No perturbation_specs keys match the chosen combo params "
            f"(chosen keys: {sorted(chosen_view)}; "
            f"configured axes: {sorted(ps_cfg.perturbation_specs)})."
        )
    spec_keys = tuple(active_specs)
    neighbor_views = build_perturbation_neighbors(
        {str(k): v for k, v in chosen_view.items()},
        active_specs,
    )
    neighbors = [
        _merge_perturbation_param_view(full_chosen, view) for view in neighbor_views
    ]

    center_metric = _metric_for_params(config, full_chosen)
    run_records: list[PerturbationRunRecord] = [
        PerturbationRunRecord(
            role="center",
            param_changed="",
            direction="",
            metric=center_metric,
            params=dict(full_chosen),
        )
    ]
    neighbor_metrics: list[float] = []
    for neighbor in neighbors:
        metric = _metric_for_params(config, neighbor)
        param_changed, direction = _neighbor_delta(
            full_chosen, neighbor, spec_keys=spec_keys
        )
        neighbor_metrics.append(metric)
        run_records.append(
            PerturbationRunRecord(
                role="neighbor",
                param_changed=param_changed,
                direction=direction,
                metric=metric,
                params=dict(neighbor),
            )
        )

    metric_floor = 2.0 if ps_cfg.metric_floor is None else float(ps_cfg.metric_floor)
    result = aggregate_perturbation_results(
        chosen_params=chosen_view,
        chosen_metric=center_metric,
        perturbed_metrics=neighbor_metrics,
        metric_floor=metric_floor,
    )
    artifact_paths = write_perturbation_artifacts(
        result,
        tuple(run_records),
        perturbation_metric=ps_cfg.perturbation_metric,
    )
    return PerturbationPipelineResult(
        result=result,
        runs=tuple(run_records),
        **artifact_paths,
    )


def write_perturbation_artifacts(
    result: PerturbationTestResult,
    runs: Sequence[PerturbationRunRecord],
    *,
    perturbation_metric: str,
    visualization_parent_dir: Path | None = None,
) -> dict[str, Path]:
    """Write perturbation JSON, CSV, and markdown under the visualization dir."""

    out_dir = (
        visualization_parent_dir / "visualization"
        if visualization_parent_dir is not None
        else canonical_in_sample_visualization_dir()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report_json = out_dir / "perturbation_report.json"
    payload = {
        **result.to_json_dict(),
        "perturbation_metric": perturbation_metric,
        "neighbor_runs": [
            {
                "role": run.role,
                "param_changed": run.param_changed,
                "direction": run.direction,
                "metric": run.metric,
                "params": run.params,
            }
            for run in runs
        ],
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    runs_csv = out_dir / "perturbation_runs.csv"
    rows = [
        {
            "role": run.role,
            "param_changed": run.param_changed,
            "direction": run.direction,
            "metric": run.metric,
            **{f"param_{key}": value for key, value in sorted(run.params.items())},
        }
        for run in runs
    ]
    pd.DataFrame(rows).to_csv(runs_csv, index=False)

    summary_md = out_dir / "perturbation_summary.md"
    summary_md.write_text(_perturbation_summary_markdown(result, runs), encoding="utf-8")

    return {
        "report_json": report_json,
        "runs_csv": runs_csv,
        "summary_md": summary_md,
    }


def _perturbation_summary_markdown(
    result: PerturbationTestResult,
    runs: Sequence[PerturbationRunRecord],
) -> str:
    status = "PASS" if result.passed else "FAIL"
    lines = [
        "# Min-step parameter perturbation",
        "",
        "Axis-aligned ±min_step neighbours around the robustness-selected combination.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Peak | {result.peak_metric:.4f} |",
        f"| Median (neighbours) | {result.median_metric:.4f} |",
        f"| P10 | {result.p10_metric:.4f} |",
        f"| P90 | {result.p90_metric:.4f} |",
        f"| Optimism bias | {result.optimism_bias:+.4f} |",
        f"| Stability ratio | {result.stability_ratio:.4f} |",
        f"| Floor | {result.metric_floor:.4f} |",
        f"| Result | **{status}** |",
        "",
        result.interpretation,
        "",
        "## Neighbour runs",
        "",
        "| Role | Param | Direction | Metric |",
        "|---|---|---|---:|",
    ]
    lines.extend(
        f"| {run.role} | {run.param_changed or '—'} | {run.direction or '—'} | {run.metric:.4f} |"
        for run in runs
        if run.role == "neighbor"
    )
    lines.append("")
    lines.append(f"Total neighbour evaluations: {result.n_perturbed}")
    return "\n".join(lines)
