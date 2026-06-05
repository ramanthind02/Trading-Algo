"""Pass 1 — ATR% vol-regime binning (runs before filter-gate exploration)."""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from research.feature.binning.config import BinningResearchConfig, load_binning_research_config
from research.feature.binning.pipeline import run_continuous_binning_phase
from research.feature.config import ResearchConfig
from research.feature.exploration.filter_gate_catalog import exploration_pass1_vol_regime_enabled
from research.feature.in_sample.data_loader import expand_bias_specs, first_bias_spec

_FILTER_GATE_MODULES = frozenset({"filter_gate", "filter_gate_entry_only"})


def exploration_includes_filter_gate(config: ResearchConfig) -> bool:
    """True when Pass 1 ATR% decile chart should run (auto gates or legacy filter catalog)."""
    return exploration_pass1_vol_regime_enabled(config)


def _scalarize_params(params: dict[str, Any]) -> dict[str, Any]:
    """One combo from a grid spec (first list element per axis)."""
    return {
        key: (value[0] if isinstance(value, list) and value else value)
        for key, value in params.items()
    }


def _ungated_signal_module(spec: dict[str, Any]) -> str:
    module = str(spec.get("module_name", ""))
    if module in _FILTER_GATE_MODULES:
        params = spec.get("params", {})
        if isinstance(params, dict) and "signal_module" in params:
            return str(params["signal_module"])
    return module


def _pass1_strategy_bias_spec(config: ResearchConfig) -> dict[str, Any]:
    """Ungated signal, single combo: eval signal leg when modules align, else first grid point."""
    raw = copy.deepcopy(first_bias_spec(config.bias_spec))
    eval_spec = copy.deepcopy(config.eval_bias_spec)
    if _ungated_signal_module(raw) == _ungated_signal_module(eval_spec):
        if str(eval_spec.get("module_name", "")) in _FILTER_GATE_MODULES:
            eval_params = eval_spec.get("params", {})
            if isinstance(eval_params, dict) and "signal_params" in eval_params:
                return {
                    "module_name": str(eval_params["signal_module"]),
                    "timeframes": list(eval_spec.get("timeframes", raw.get("timeframes", []))),
                    "params": copy.deepcopy(eval_params["signal_params"]),
                }
        return eval_spec
    return {
        **raw,
        "params": _scalarize_params(dict(raw.get("params", {}))),
    }


def pass1_strategy_bias_spec_expands_to_one_combo(config: ResearchConfig) -> bool:
    """Guard for binning Pass 1: ``strategy_bias_spec`` must be a single expanded combo."""
    return len(expand_bias_specs(_pass1_strategy_bias_spec(config))) == 1


def binning_config_for_exploration(config: ResearchConfig) -> BinningResearchConfig:
    """Align Pass 1 binning window, tickers, and timeframe with the exploration train slice."""
    train_start, train_end = config.training_window_bounds
    gates = config.exploration_filter_gates
    atr_period = gates.atr_period if gates is not None else 32
    base = load_binning_research_config()
    atr_bias_spec: dict[str, Any] = {
        "module_name": "atr",
        "timeframes": [config.timeframe],
        "params": {"period": atr_period},
    }
    return replace(
        base,
        tickers=list(config.tickers),
        start=train_start,
        end=train_end,
        timeframe=config.timeframe,
        target_col="log_return",
        bias_spec=atr_bias_spec,
        feature_col_substr="atrPct",
        strategy_bias_spec=_pass1_strategy_bias_spec(config),
    )


def run_pass1_vol_regime_binning(config: ResearchConfig) -> dict[str, int | str]:
    """Run ATR% decile binning and write ``atr_pct_decile_chart.png`` to visualization/matplotlib."""
    binning_config = binning_config_for_exploration(config)
    train_start, train_end = config.training_window_bounds
    ticker_names = [ticker.name for ticker in config.tickers]
    print(
        f"\nPass 1 — ATR% vol-regime binning "
        f"({train_start.date()} → {train_end.date()}, tickers={ticker_names})..."
    )
    summary = run_continuous_binning_phase(binning_config)
    print(
        "Pass 1 complete. Decile chart: "
        "feature_research/in_sample/results/visualization/matplotlib/atr_pct_decile_chart.png"
    )
    return summary
