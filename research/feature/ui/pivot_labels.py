"""Human-readable labels for the parameter-sensitivity pivot explorer."""
from __future__ import annotations

_GATE_FIELD_LABELS: dict[str, str] = {
    "filter_family": "Filter family",
    "gate_mode": "Gate mode",
    "gate_type": "Gate (A/B/C)",
    "vol_max_rank": "Vol cap (rank)",
    "filter_detail": "Filter detail",
    "filter_module": "Filter module",
    "signal_module": "Signal module",
    "ticker": "Ticker",
    "param_combo_label": "Parameter combo",
    "research_display_label": "Combo label",
    "feature_name": "Feature",
}

_METRIC_LABELS: dict[str, str] = {
    "t_stat": "T-stat",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "n_observations": "Observations",
    "n_nonzero_signal": "Non-zero signal days",
    "trade_reduction_pct": "Trade reduction %",
}


def _humanize_token(raw: str) -> str:
    return " ".join(part.capitalize() for part in raw.replace("-", "_").split("_") if part)


def param_field_display_label(field: str) -> str:
    """Map an internal pivot field key to a researcher-facing label."""
    if field in _GATE_FIELD_LABELS:
        return _GATE_FIELD_LABELS[field]
    if field.startswith("s_"):
        return f"Signal · {_humanize_token(field[2:])}"
    if field.startswith("f_"):
        return f"Filter · {_humanize_token(field[2:])}"
    if field.startswith("a_"):
        return f"Leg A · {_humanize_token(field[2:])}"
    if field.startswith("b_"):
        return f"Leg B · {_humanize_token(field[2:])}"
    return _humanize_token(field)


def param_field_group(field: str) -> str:
    """Group pivot fields for optgroup rendering in the workspace UI."""
    if field in {"filter_family", "gate_mode", "gate_type", "vol_max_rank", "filter_detail"}:
        return "Gate & setup"
    if field.startswith("s_") or field == "signal_module":
        return "Signal parameters"
    if field.startswith("f_") or field == "filter_module":
        return "Filter parameters"
    if field.startswith(("a_", "b_")):
        return "Dual-signal legs"
    if field == "ticker":
        return "Market slice"
    return "Strategy parameters"


def metric_display_label(metric: str) -> str:
    """Map a metric column name to a display label."""
    return _METRIC_LABELS.get(metric, _humanize_token(metric))
