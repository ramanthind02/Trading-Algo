"""Combo-count and effective-search diagnostics for the research workspace UI."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from research.feature.config import ResearchConfig
from research.feature.exploration.filter_gate_catalog import (
    exploration_filter_gate_combo_multiplier,
    resolved_exploration_bias_spec,
)
from research.feature.in_sample.data_loader import expand_bias_specs, param_combo_label
from research.feature.ui.contracts import ComboDiagnostics


def build_combo_diagnostics(config: ResearchConfig) -> ComboDiagnostics:
    """Return nominal and effective combo counts for the configured exploration grid."""

    expanded_specs = expand_bias_specs(resolved_exploration_bias_spec(config))
    raw_combo_count = len(expanded_specs)
    gates = config.exploration_filter_gates
    if gates.enabled and gates.scope == "winning_signal_only":
        raw_combo_count += exploration_filter_gate_combo_multiplier(
            rank_perturbation_pct=gates.rank_perturbation_pct
        )
    evaluation_combo_label = param_combo_label(dict(config.eval_bias_spec.get("params", {})))
    if raw_combo_count == 0:
        return ComboDiagnostics(
            raw_combo_count=0,
            loaded_combo_count=0,
            effective_combo_count=None,
            evaluation_combo_label=evaluation_combo_label,
            notes=("No exploration combinations were generated.",),
        )

    report_based = _report_backed_combo_diagnostics(
        config=config,
        raw_combo_count=raw_combo_count,
        evaluation_combo_label=evaluation_combo_label,
    )
    if report_based is not None:
        return report_based

    if raw_combo_count == 1:
        return ComboDiagnostics(
            raw_combo_count=raw_combo_count,
            loaded_combo_count=1,
            effective_combo_count=1.0,
            evaluation_combo_label=evaluation_combo_label,
            surface_interpretation="Single combination only, so effective combos equal raw combos.",
            notes=("Single configured combination only.",),
        )
    return ComboDiagnostics(
        raw_combo_count=raw_combo_count,
        loaded_combo_count=0,
        effective_combo_count=None,
        evaluation_combo_label=evaluation_combo_label,
        notes=(
            "Effective combo count appears after exploration robustness writes "
            "`robustness_report.json` for the current configured search space.",
        ),
    )


def _report_backed_combo_diagnostics(
    *,
    config: ResearchConfig,
    raw_combo_count: int,
    evaluation_combo_label: str,
) -> ComboDiagnostics | None:
    report_path = Path(config.reports_dir) / "robustness_report.json"
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_n_effective = payload.get("n_effective", {})
        effective_value = (
            float(report_n_effective.get("n_effective"))
            if isinstance(report_n_effective, Mapping) and report_n_effective.get("n_effective") is not None
            else None
        )
        report_combo_count = int(payload.get("n_combinations", raw_combo_count))
        surface_interpretation = (
            str(report_n_effective.get("surface_label"))
            if isinstance(report_n_effective, Mapping) and report_n_effective.get("surface_label") is not None
            else None
        )
        return ComboDiagnostics(
            raw_combo_count=raw_combo_count,
            loaded_combo_count=report_combo_count,
            effective_combo_count=effective_value,
            evaluation_combo_label=evaluation_combo_label,
            surface_interpretation=surface_interpretation,
            notes=("Effective combo count loaded from the current exploration robustness report.",),
        )
    except Exception:
        return None

