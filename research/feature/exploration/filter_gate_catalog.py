"""Standard filter-gate branches appended to signal-only exploration specs."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from research.feature._internal.bias_spec_catalog import first_bias_spec

_FILTER_GATE_MODULES = frozenset({"filter_gate", "filter_gate_entry_only"})
_GATE_MODULE_NAMES: list[str] = ["filter_gate_entry_only", "filter_gate"]


def atr_rank_fraction_grid(center: float, pct: float = 0.3) -> list[float]:
    """Three ``max_rank_fraction`` values at ±``pct`` around ``center``, clamped to (0, 1]."""
    if not 0.0 < center <= 1.0:
        raise ValueError("center must be in (0, 1].")
    if not 0.0 <= pct < 1.0:
        raise ValueError("pct must be in [0, 1).")
    candidates = (
        center * (1.0 - pct),
        center,
        center * (1.0 + pct),
    )
    return sorted({max(min(float(value), 1.0), 1e-6) for value in candidates})


def exploration_filter_gate_combo_multiplier(*, rank_perturbation_pct: float = 0.3) -> int:
    """Branch multiplier: baseline + 2 SMA gates + 2×rank_grid low + 2×rank_grid high."""
    rank_grid_size = len(atr_rank_fraction_grid(0.2, rank_perturbation_pct))
    return 1 + 2 + (2 * rank_grid_size) + (2 * rank_grid_size)


@dataclass(frozen=True)
class ExplorationFilterGatesConfig:
    """Hard-coded exploration filters (SMA trend + ATR% low/high); validation stays manual."""

    enabled: bool = True
    scope: Literal["full_signal_grid", "winning_signal_only"] = "winning_signal_only"
    rank_perturbation_pct: float = 0.3
    sma_period: int = 252
    atr_period: int = 32
    atr_lookback: int = 252
    atr_low_rank_center: float = 0.2
    atr_high_rank_center: float = 0.6

    def __post_init__(self) -> None:
        if self.sma_period < 2:
            raise ValueError("sma_period must be >= 2")
        if self.atr_period < 1 or self.atr_lookback < 2:
            raise ValueError("atr_period must be >= 1 and atr_lookback >= 2")
        if not 0.0 < self.rank_perturbation_pct < 1.0:
            raise ValueError("rank_perturbation_pct must be in (0, 1)")
        if self.scope not in ("full_signal_grid", "winning_signal_only"):
            raise ValueError(
                "scope must be 'full_signal_grid' or 'winning_signal_only', "
                f"got {self.scope!r}"
            )


def _validate_signal_spec(signal_spec: dict[str, Any]) -> tuple[str, dict[str, Any], list[Any]]:
    module_name = signal_spec.get("module_name")
    if isinstance(module_name, list):
        raise ValueError(
            "build_exploration_catalog_with_filter_gates expects a single signal module; "
            "got list-valued module_name (use a catalog list at the load_config level)."
        )
    module = str(module_name or "")
    if module in _FILTER_GATE_MODULES:
        raise ValueError(
            f"signal_spec must not already be a filter gate module, got {module!r}"
        )
    if "filter_module" in signal_spec.get("params", {}):
        raise ValueError("signal_spec params must be the signal leg only, not filter_gate shape")
    signal_params = copy.deepcopy(signal_spec.get("params", {}))
    timeframes = copy.deepcopy(signal_spec.get("timeframes", []))
    return module, signal_params, timeframes


def _filter_gate_branch(
    *,
    timeframes: list[Any],
    filter_module: str,
    filter_params: dict[str, Any],
    signal_module: str,
    signal_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "module_name": list(_GATE_MODULE_NAMES),
        "timeframes": timeframes,
        "params": {
            "filter_module": filter_module,
            "filter_params": filter_params,
            "signal_module": signal_module,
            "signal_params": signal_params,
        },
    }


def build_exploration_catalog_with_filter_gates(
    signal_spec: dict[str, Any],
    *,
    gate_config: ExplorationFilterGatesConfig | None = None,
) -> list[dict[str, Any]]:
    """Return [baseline, SMA252 B/C, ATR low B/C, ATR high B/C] sharing ``signal_spec``'s param grid."""
    cfg = gate_config or ExplorationFilterGatesConfig()
    if not cfg.enabled:
        return [copy.deepcopy(signal_spec)]

    signal_module, signal_params, timeframes = _validate_signal_spec(signal_spec)
    rank_grid_low = atr_rank_fraction_grid(cfg.atr_low_rank_center, cfg.rank_perturbation_pct)
    rank_grid_high = atr_rank_fraction_grid(cfg.atr_high_rank_center, cfg.rank_perturbation_pct)

    return [
        copy.deepcopy(signal_spec),
        _filter_gate_branch(
            timeframes=timeframes,
            filter_module="sma_above_filter",
            filter_params={"period": cfg.sma_period},
            signal_module=signal_module,
            signal_params=copy.deepcopy(signal_params),
        ),
        _filter_gate_branch(
            timeframes=timeframes,
            filter_module="atr_percentile_filter",
            filter_params={
                "atr_period": cfg.atr_period,
                "lookback": cfg.atr_lookback,
                "rank_metric": "atr_pct",
                "percentile_tail": "low",
                "max_rank_fraction": rank_grid_low,
            },
            signal_module=signal_module,
            signal_params=copy.deepcopy(signal_params),
        ),
        _filter_gate_branch(
            timeframes=timeframes,
            filter_module="atr_percentile_filter",
            filter_params={
                "atr_period": cfg.atr_period,
                "lookback": cfg.atr_lookback,
                "rank_metric": "atr_pct",
                "percentile_tail": "high",
                "max_rank_fraction": rank_grid_high,
            },
            signal_module=signal_module,
            signal_params=copy.deepcopy(signal_params),
        ),
    ]


def _scalarize_params_dict(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (value[0] if isinstance(value, list) and value else value)
        for key, value in params.items()
    }


def scalar_signal_bias_spec_from_chosen(
    config: Any,
    chosen_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Single-combo ungated signal spec from robustness ``best_combination.params``."""
    from research.feature.in_sample.data_loader import BIAS_MODULE_COMBO_KEY

    template = copy.deepcopy(first_bias_spec(config.bias_spec))
    flat = dict(chosen_params)
    module = str(flat.get(BIAS_MODULE_COMBO_KEY, "") or "")
    skip_meta = frozenset(
        {
            BIAS_MODULE_COMBO_KEY,
            "bias_composite_module",
            "filter_module",
            "filter_params",
            "signal_module",
            "signal_params",
        }
    )
    if module in _FILTER_GATE_MODULES:
        signal_module = str(flat.get("signal_module") or template.get("module_name", ""))
        nested = flat.get("signal_params")
        if not isinstance(nested, Mapping):
            raise ValueError(
                "filter-gate winner params must include signal_params mapping"
            )
        template["module_name"] = signal_module
        template["params"] = _scalarize_params_dict(dict(nested))
    else:
        template["params"] = _scalarize_params_dict(
            {
                key: value
                for key, value in flat.items()
                if key not in skip_meta and not str(key).startswith("f_")
            }
        )
    return template


def resolved_filter_gate_bias_spec(
    config: Any,
    chosen_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Filter A/B/C catalog for one winning signal combo (~15 expanded combos)."""
    gates = getattr(config, "exploration_filter_gates", ExplorationFilterGatesConfig())
    if not gates.enabled:
        raise ValueError("resolved_filter_gate_bias_spec requires exploration_filter_gates.enabled")
    signal_spec = scalar_signal_bias_spec_from_chosen(config, chosen_params)
    return build_exploration_catalog_with_filter_gates(signal_spec, gate_config=gates)


def resolve_exploration_winner_params(
    config: Any,
    *,
    robustness_report: Any | None = None,
) -> dict[str, Any]:
    """Params for the best ungated signal combo (robustness winner or param-sensitivity leader)."""
    import pandas as pd

    from research.feature.in_sample.data_loader import (
        enrich_param_combo_with_module,
        expand_bias_specs,
        expanded_spec_combo_label,
    )
    from research.feature.shared.visualization_paths import canonical_in_sample_visualization_dir

    if robustness_report is not None:
        return dict(robustness_report.best_combination.params)

    expanded = expand_bias_specs(config.bias_spec)
    if len(expanded) == 1:
        spec = expanded[0]
        return enrich_param_combo_with_module(
            spec["params"],
            spec.get("module_name"),
        )

    viz_dir = canonical_in_sample_visualization_dir()
    ps_path = viz_dir / "param_sensitivity.csv"
    if not ps_path.is_file():
        raise ValueError(
            "Filter-gate follow-up requires robustness or an existing param_sensitivity.csv"
        )
    frame = pd.read_csv(ps_path)
    robustness_cfg = getattr(config, "robustness", None)
    selection_metric = (
        getattr(robustness_cfg, "selection_metric", None) if robustness_cfg else None
    )
    metric_col = str(getattr(selection_metric, "builtin", None) or "t_stat")
    if metric_col not in frame.columns:
        metric_col = "t_stat"
    if frame.empty or "param_combo_label" not in frame.columns:
        raise ValueError("param_sensitivity.csv is empty or missing param_combo_label")
    best_label = str(
        frame.sort_values(metric_col, ascending=False).iloc[0]["param_combo_label"]
    )
    for spec in expanded:
        if expanded_spec_combo_label(spec) == best_label:
            return enrich_param_combo_with_module(
                spec["params"],
                spec.get("module_name"),
            )
    raise ValueError(f"Winner label {best_label!r} not found in exploration bias grid")


def resolved_exploration_bias_spec(
    config: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    """In-sample exploration spec: signal grid, or 15× grid when scope is ``full_signal_grid``."""
    raw = config.bias_spec
    gates = getattr(config, "exploration_filter_gates", ExplorationFilterGatesConfig())
    if not gates.enabled or gates.scope == "winning_signal_only":
        return raw
    if isinstance(raw, list):
        if len(raw) > 1 or str(first_bias_spec(raw).get("module_name", "")) in _FILTER_GATE_MODULES:
            return raw
        base = first_bias_spec(raw)
    else:
        if str(raw.get("module_name", "")) in _FILTER_GATE_MODULES:
            return raw
        base = raw
    return build_exploration_catalog_with_filter_gates(base, gate_config=gates)


def exploration_pass1_vol_regime_enabled(config: Any) -> bool:
    """Run ATR% decile chart when auto gates are on or catalog already has filter gates."""
    gates = getattr(config, "exploration_filter_gates", None)
    if gates is not None and gates.enabled:
        return True
    from research.feature.filter_research_labels import is_filter_exploration_modules
    from research.feature.in_sample.data_loader import expand_bias_specs

    expanded = expand_bias_specs(config.bias_spec)
    modules = {str(spec.get("module_name", "")) for spec in expanded}
    return is_filter_exploration_modules(modules)
