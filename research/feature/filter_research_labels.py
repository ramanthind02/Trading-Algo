"""Human-readable labels and pivot dimensions for filter-gate exploration (A/B/C)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from research.feature.binning.transforms import flatten_params_for_combo_long_table
from research.feature.in_sample.data_loader import (
    BIAS_MODULE_COMBO_KEY,
    expanded_combo_param_value,
    param_combo_display_label,
)

_FILTER_GATE_MODULES = frozenset({"filter_gate", "filter_gate_entry_only"})


def is_filter_exploration_modules(module_names: set[str]) -> bool:
    return bool(module_names & _FILTER_GATE_MODULES)


def is_filter_exploration_params(params: Mapping[str, Any]) -> bool:
    module = str(params.get(BIAS_MODULE_COMBO_KEY, "") or "")
    if module in _FILTER_GATE_MODULES:
        return True
    return "filter_module" in params or "filter_params" in params


def _flat_params(params: Mapping[str, Any]) -> dict[str, Any]:
    flat = flatten_params_for_combo_long_table(params)
    if BIAS_MODULE_COMBO_KEY in params:
        flat[BIAS_MODULE_COMBO_KEY] = params[BIAS_MODULE_COMBO_KEY]
    return flat


def filter_exploration_semantics(
    params: Mapping[str, Any],
    *,
    label: str | None = None,
) -> dict[str, str]:
    """Derive stable pivot / table columns for one exploration combo."""
    flat = _flat_params(params)
    module = str(
        flat.get(BIAS_MODULE_COMBO_KEY, "")
        or (label.split("__", 1)[0] if label and "__" in label else "")
    )
    gate_type = "A"
    gate_mode = "n/a"
    if module == "filter_gate_entry_only":
        gate_type, gate_mode = "B", "entry_only"
    elif module == "filter_gate":
        gate_type, gate_mode = "C", "full_gate"

    filter_module = str(flat.get("filter_module", "") or "")
    if gate_type == "A" or not filter_module:
        return {
            "gate_type": gate_type,
            "filter_family": "baseline",
            "gate_mode": gate_mode,
            "vol_max_rank": "n/a",
            "filter_detail": "unfiltered cumrsi",
        }

    if filter_module == "sma_above_filter":
        period = flat.get("f_period", flat.get("period", "?"))
        return {
            "gate_type": gate_type,
            "filter_family": "trend",
            "gate_mode": gate_mode,
            "vol_max_rank": "n/a",
            "filter_detail": f"SMA({period}) price above",
        }

    if filter_module == "atr_percentile_filter":
        rank = flat.get("f_max_rank_fraction", "?")
        tail = flat.get("f_percentile_tail", "high")
        atr_period = flat.get("f_atr_period", 14)
        lookback = flat.get("f_lookback", 252)
        return {
            "gate_type": gate_type,
            "filter_family": "vol",
            "gate_mode": gate_mode,
            "vol_max_rank": str(rank),
            "filter_detail": (
                f"ATR({atr_period}) {tail} tail ≤{rank} rank · {lookback}d lookback"
            ),
        }

    return {
        "gate_type": gate_type,
        "filter_family": filter_module or "other",
        "gate_mode": gate_mode,
        "vol_max_rank": "n/a",
        "filter_detail": filter_module,
    }


def research_display_label(
    params: Mapping[str, Any],
    *,
    label: str | None = None,
) -> str:
    """Readable combo name for robustness tables, charts, and filter summaries."""
    if not params:
        return "unknown"

    semantics = filter_exploration_semantics(params, label=label)
    gate = semantics["gate_type"]
    if semantics["filter_family"] == "baseline":
        signal = param_combo_display_label(
            {k: v for k, v in params.items() if k != BIAS_MODULE_COMBO_KEY}
        )
        return f"A: Baseline · {signal}" if signal else "A: Baseline (unfiltered)"

    detail = semantics["filter_detail"]
    mode = semantics["gate_mode"]
    if gate == "B":
        return f"B: {detail} · entry-only"
    if gate == "C":
        return f"C: {detail} · full gate"
    return f"{gate}: {detail}"


def build_filter_exploration_long_pairs(
    label: str,
    params: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Long-param row for pivot: gate semantics plus flattened ``f_*`` / ``s_*`` feature params."""
    semantics = filter_exploration_semantics(params, label=label)
    flat = _flat_params(params)
    return [(label, {**flat, **semantics})]


def permutation_combo_display_name(params: dict[str, Any] | None) -> str:
    """Human-readable label; filter exploration uses :func:`research_display_label`."""
    if not params:
        return "unknown"
    if is_filter_exploration_params(params):
        return research_display_label(params)
    scalar_like = all(
        v is None or isinstance(v, (bool, int, float, str)) for v in params.values()
    )
    if scalar_like:
        return param_combo_display_label(params)
    return research_display_label(params)
