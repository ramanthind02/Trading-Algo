"""Artifact discovery and preview helpers for the unified research workspace."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.feature.config import ResearchConfig
from research.feature.shared import FeatureResearchPhase
from research.feature.ui.artifact_catalog import (
    build_phase_report_sections,
    discover_artifact_records,
    ensure_phase_visualization_reports,
    group_artifact_records,
    phase_discovery_roots,
)
from research.feature.shared.visualization_paths import canonical_in_sample_visualization_dir
from research.feature.ui.workspace_manifest import (
    MANIFEST_FILENAME,
    exploration_viz_matches_config,
    read_workspace_manifest,
    write_exploration_manifests,
)
from research.feature.ui.pivot_data import pivot_explorer_metadata_for_phase
from research.feature.ui.planner import resolve_ui_config

_REPO_ROOT = Path(__file__).resolve().parents[3]


def build_workspace_view(
    config: ResearchConfig,
    request,
    *,
    current_job: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one app-friendly workspace view for the current UI selection."""

    configured, _selection = resolve_ui_config(config, request)
    phase_views = [
        _phase_workspace_view(configured, phase)
        for phase in FeatureResearchPhase
    ]
    return {
        "selected_phase": request.phase.value,
        "job": current_job,
        "phases": phase_views,
        "vault_commit": _vault_commit_view(config, request),
    }


def _vault_commit_view(config: ResearchConfig, request) -> dict[str, object]:
    from research.feature.ui.vault_save import build_vault_commit_view

    return build_vault_commit_view(config, request)


def load_artifact_preview(relative_path: str) -> dict[str, object]:
    """Return a lightweight preview payload for one workspace artifact."""

    from research.workspace.preview import load_artifact_preview as _load_preview

    return _load_preview(_REPO_ROOT, relative_path)


def resolve_workspace_artifact_path(relative_path: str) -> Path:
    """Resolve a repo-relative artifact path and keep access inside the repo."""

    from research.workspace.preview import resolve_artifact_path

    return resolve_artifact_path(_REPO_ROOT, relative_path)


def _phase_workspace_view(
    config: ResearchConfig,
    phase: FeatureResearchPhase,
) -> dict[str, object]:
    if phase is FeatureResearchPhase.EXPLORATION:
        _maybe_refresh_exploration_plots(config)
    roots = phase_discovery_roots(config, phase)
    artifacts = discover_artifact_records(
        roots,
        phase=phase,
        relative_to=_REPO_ROOT,
    )
    grouped_artifacts = group_artifact_records(artifacts, phase)
    report_sections, default_section_id = build_phase_report_sections(grouped_artifacts, phase)
    return {
        "phase": phase.value,
        "label": phase.value.replace("_", " ").title(),
        "summary_cards": _summary_cards(phase, artifacts),
        "artifact_counts": dict(Counter(str(artifact["kind"]) for artifact in artifacts)),
        "category_counts": dict(Counter(str(artifact["category"]) for artifact in artifacts)),
        "report_sections": report_sections,
        "default_section_id": default_section_id,
        "has_results": bool(report_sections),
        "pivot_explorer": pivot_explorer_metadata_for_phase(config, phase),
    }


def _summary_cards(
    phase: FeatureResearchPhase,
    artifacts: list[dict[str, object]],
) -> list[dict[str, str]]:
    cards = [
        {"label": "Artifacts", "value": str(len(artifacts))},
    ]
    path_lookup = {
        str(artifact["relative_path"]): artifact
        for artifact in artifacts
    }
    plot_count = sum(
        1
        for artifact in artifacts
        if artifact["kind"] in {"html", "image"}
    )
    if plot_count:
        cards.append({"label": "Plots & reports", "value": str(plot_count)})

    if phase is FeatureResearchPhase.EXPLORATION:
        robustness_path = _first_path_ending(path_lookup, "robustness_report.json")
        if robustness_path is not None:
            payload = json.loads(resolve_workspace_artifact_path(robustness_path).read_text(encoding="utf-8"))
            n_effective = payload.get("n_effective", {})
            rolling = payload.get("rolling_is", {})
            stability = payload.get("stability_chart", rolling)
            full_grid = payload.get("full_grid_permutation")
            cards.extend(
                [
                    {"label": "Module", "value": str(payload.get("feature_name", "unknown"))},
                    {"label": "Raw combos", "value": str(payload.get("n_combinations", "unknown"))},
                    {"label": "Effective combos", "value": f"{float(n_effective.get('n_effective', 0.0)):.2f}"},
                    {
                        "label": "6-mo rolling CUSUM",
                        "value": (
                            f"{float(stability.get('cusum_statistic', 0.0)):.2f} / "
                            f"{float(stability.get('cusum_critical_value', 1.36)):.2f}"
                        ),
                    },
                ]
            )
            if isinstance(full_grid, dict) and full_grid.get("p_value") is not None:
                cards.append(
                    {
                        "label": "Full-grid permutation p",
                        "value": f"{float(full_grid['p_value']):.4f}",
                    }
                )
        perturbation_path = _first_path_ending(path_lookup, "perturbation_report.json")
        if perturbation_path is not None:
            perturbation = json.loads(
                resolve_workspace_artifact_path(perturbation_path).read_text(encoding="utf-8")
            )
            cards.extend(
                [
                    {
                        "label": "Perturbation peak",
                        "value": f"{float(perturbation.get('peak_metric', 0.0)):.2f}",
                    },
                    {
                        "label": "Perturbation median",
                        "value": f"{float(perturbation.get('median_metric', 0.0)):.2f}",
                    },
                    {
                        "label": "Stability ratio",
                        "value": f"{float(perturbation.get('stability_ratio', 0.0)):.2f}",
                    },
                    {
                        "label": "Perturbation",
                        "value": "PASS" if perturbation.get("passed") else "FAIL",
                    },
                ]
            )
        return cards

    if phase is FeatureResearchPhase.VALIDATION:
        validation_robustness_path = _first_path_ending(
            path_lookup,
            "validation_robustness_report.json",
        )
        if validation_robustness_path is not None:
            payload = json.loads(
                resolve_workspace_artifact_path(validation_robustness_path).read_text(
                    encoding="utf-8"
                )
            )
            sharpe = payload.get("sharpe_comparison", {})
            cusum = payload.get("cusum", {})
            bands = payload.get("equity_curve_bands", {})
            rolling = payload.get("rolling_sharpe_zscore", {})
            rank = payload.get("rank_correlation", {})
            ci_is = sharpe.get("ci_is", {})
            ci_val = sharpe.get("ci_val", {})
            cards.extend(
                [
                    {
                        "label": "IS Sharpe",
                        "value": (
                            f"{float(sharpe.get('sr_is', 0.0)):.2f} "
                            f"[{float(ci_is.get('lower', 0.0)):.2f}, {float(ci_is.get('upper', 0.0)):.2f}]"
                        ),
                    },
                    {
                        "label": "Val Sharpe",
                        "value": (
                            f"{float(sharpe.get('sr_val', 0.0)):.2f} "
                            f"[{float(ci_val.get('lower', 0.0)):.2f}, {float(ci_val.get('upper', 0.0)):.2f}]"
                        ),
                    },
                    {
                        "label": "Degradation ratio",
                        "value": f"{float(sharpe.get('degradation_ratio', 0.0)):.2f}",
                    },
                    {
                        "label": "CI overlap",
                        "value": "Yes" if sharpe.get("ci_overlap") else "No",
                    },
                    {
                        "label": "CUSUM",
                        "value": "PASS" if cusum.get("passed") else "FAIL",
                    },
                    {
                        "label": "Equity bands",
                        "value": (
                            f"{'PASS' if bands.get('passed') else 'FAIL'} "
                            f"({float(bands.get('fraction_below_lower', 0.0)):.0%} below lower)"
                        ),
                    },
                    {
                        "label": "Rolling SR z-score",
                        "value": (
                            f"{'PASS' if rolling.get('passed') else 'FAIL'} "
                            f"({float(rolling.get('fraction_below_threshold', 0.0)):.0%} below z)"
                        ),
                    },
                    {
                        "label": "Rank correlation ρ",
                        "value": f"{float(rank.get('spearman_rho', 0.0)):.2f}",
                    },
                    {
                        "label": "Overall",
                        "value": "PASS" if payload.get("all_passed") else "FAIL",
                    },
                ]
            )
        cards.extend(_portfolio_gate_summary_cards(path_lookup))

    if phase is FeatureResearchPhase.PORTFOLIO_ADDITION:
        cards.extend(_portfolio_gate_summary_cards(path_lookup))
        return cards

    report_path = _first_path_ending(path_lookup, "report.json")
    if report_path is not None:
        payload = json.loads(resolve_workspace_artifact_path(report_path).read_text(encoding="utf-8"))
        tearsheets = payload.get("tearsheet_files", [])
        cards.extend(
            [
                {"label": "Module", "value": str(payload.get("module_name", "unknown"))},
                {"label": "Objective", "value": str(payload.get("objective_metric_name", "unknown"))},
                {"label": "Tearsheets", "value": str(len(tearsheets))},
            ]
        )
    equity_path = _first_path_ending(path_lookup, "equity_curve_validation_only.csv")
    if equity_path is not None:
        frame = pd.read_csv(resolve_workspace_artifact_path(equity_path))
        if "cumulative_strategy_return" in frame.columns and not frame.empty:
            cards.append(
                {
                    "label": "Latest cumulative return",
                    "value": f"{float(frame['cumulative_strategy_return'].iloc[-1]):.4f}",
                }
            )
    return cards


def _maybe_refresh_exploration_plots(config: ResearchConfig) -> None:
    """Stamp a fresh manifest and regenerate Matplotlib plots for the active exploration run."""

    if not hasattr(config, "eval_bias_spec"):
        return

    viz_dir = canonical_in_sample_visualization_dir()
    manifest_path = viz_dir / MANIFEST_FILENAME
    has_exploration_csvs = (viz_dir / "param_sensitivity.csv").is_file()
    manifest = read_workspace_manifest(manifest_path)
    if has_exploration_csvs and not exploration_viz_matches_config(manifest, config):
        write_exploration_manifests(config)
        manifest = read_workspace_manifest(manifest_path)
    if not exploration_viz_matches_config(manifest, config):
        return
    ensure_phase_visualization_reports(config, FeatureResearchPhase.EXPLORATION)


def _first_path_ending(path_lookup: dict[str, dict[str, object]], suffix: str) -> str | None:
    match = next((path for path in path_lookup if path.endswith(suffix)), None)
    return match


def _portfolio_gate_summary_cards(
    path_lookup: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    gate_path = _first_path_ending(path_lookup, "portfolio_addition_report.json")
    if gate_path is None:
        return []
    payload = json.loads(
        resolve_workspace_artifact_path(gate_path).read_text(encoding="utf-8")
    )
    if payload.get("skipped"):
        return [
            {"label": "Portfolio gate", "value": "Skipped (first strategy)"},
            {"label": "Evaluation data", "value": _portfolio_gate_eval_window(payload)},
        ]
    hurdle = payload.get("analytical_hurdle", {})
    empirical = payload.get("empirical_comparison", {})
    pairwise = payload.get("pairwise_redundancy", {})
    context = payload.get("context", {})
    drawdown = context.get("drawdown_detail", {}) if isinstance(context, dict) else {}
    ci = empirical.get("delta_sr_ci", {}) if isinstance(empirical, dict) else {}
    sleeve_label = str(payload.get("sleeve_label") or context.get("sleeve_label") or "sleeve")
    portfolio_block = payload.get("portfolio") or context.get("portfolio_gate") or {}
    port_empirical = (
        portfolio_block.get("empirical_comparison", {})
        if isinstance(portfolio_block, dict)
        else {}
    )
    return [
        {"label": "Gate scope", "value": f"Sleeve {sleeve_label} (primary)"},
        {"label": "Evaluation data", "value": _portfolio_gate_eval_window(payload)},
        {
            "label": "Pairwise max corr",
            "value": (
                f"{float(pairwise.get('max_pairwise_corr', 0.0)):.2f} "
                f"({'flagged' if pairwise.get('flagged') else 'ok'})"
            ),
        },
        {
            "label": "Effective ρ",
            "value": f"{float(hurdle.get('corr_effective', 0.0)):.2f}",
        },
        {
            "label": "DD overlap",
            "value": f"{float(drawdown.get('drawdown_overlap', hurdle.get('drawdown_overlap', 0.0))):.0%}",
        },
        {
            "label": "Hurdle margin",
            "value": f"{float(hurdle.get('margin', 0.0)):+.2f}",
        },
        {
            "label": "Sleeve ΔSR",
            "value": (
                f"{float(empirical.get('delta_sr', 0.0)):+.4f} "
                f"(≥ {float(empirical.get('delta_sr_threshold', 0.02)):.2f}) "
                f"[{float(ci.get('lower', 0.0)):+.3f}, {float(ci.get('upper', 0.0)):+.3f}]"
            ),
        },
        {
            "label": "Portfolio ΔSR",
            "value": (
                f"{float(port_empirical.get('delta_sr', 0.0)):+.4f} "
                f"(≥ {float(port_empirical.get('delta_sr_threshold', 0.02)):.2f}, context)"
            ),
        },
        {
            "label": "Weight assigned",
            "value": f"{float(empirical.get('weight_assigned', 0.0)):.2%}",
        },
        {
            "label": "Weight layer",
            "value": _format_weight_layer_summary(context),
        },
        {
            "label": "Gate result",
            "value": "PASS" if payload.get("passed") else "FAIL",
        },
    ]


def _format_weight_layer_summary(context: dict[str, object]) -> str:
    method = str(context.get("weight_layer_method", "unknown"))
    if method == "hierarchy_equal" or context.get("weight_layer_hierarchy_mode") == "asset_first":
        method = "hierarchy_equal (asset-first)"
    detail = context.get("weight_layer_detail")
    if isinstance(detail, dict):
        streams = detail.get("stream_count_without"), detail.get("stream_count_with")
        if streams[0] is not None and streams[1] is not None:
            method = f"{method} · {streams[0]}→{streams[1]} streams"
        candidate_total = detail.get("candidate_weight_total")
        if isinstance(candidate_total, (int, float)):
            method = f"{method} · candidate {float(candidate_total):.1%}"
    budgets = context.get("asset_class_budgets")
    if isinstance(budgets, dict) and budgets:
        parts = [
            f"{name} {float(weight):.0%}"
            for name, weight in sorted(budgets.items(), key=lambda item: item[0])
        ]
        return f"{method} · {' · '.join(parts)}"
    return method


def _portfolio_gate_eval_window(payload: dict[str, object]) -> str:
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        return "IS + Validation"
    start = meta.get("eval_start") or meta.get("train_start")
    end = meta.get("eval_end") or meta.get("val_end")
    if start and end:
        return f"IS + Validation {start} → {end}"
    return "IS + Validation"


