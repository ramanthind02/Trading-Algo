"""Portfolio research: defaults, run launching, and artifacts.

Thin wrapper over the existing ``research.portfolio.ui`` logic (planner + the
``PortfolioWorkspaceJobManager``), so the React app can run the portfolio pipeline on the
canonical portfolio config (the whole vault portfolio) the same way it runs feature exploration.

The portfolio config (``research/portfolio/config.py``) is also the **baseline** the feature
portfolio-addition gate scores against — see :mod:`frontend.api.feature_runs` for that link.
"""

from __future__ import annotations

from typing import Any

from frontend.api import artifacts as artifacts_mod
from frontend.api.paths import PORTFOLIO_RESULTS_ROOT, repo_relative

_MANAGER = None

# Weight-layer kwargs the UI may edit (SR-tilt knobs). Cast on the way in.
_SR_FLOAT_KEYS = ("sr_avg", "sr_p_step", "sr_min_years", "fdm_max")
_WEIGHTS_CSV = PORTFOLIO_RESULTS_ROOT / "weight_layer_weights_all_phases.csv"


def _manager():
    global _MANAGER
    if _MANAGER is None:
        from research.portfolio.ui.job_manager import PortfolioWorkspaceJobManager

        _MANAGER = PortfolioWorkspaceJobManager()
    return _MANAGER


def _load_config():
    from research.portfolio.config import load_config

    return load_config()


def _planner():
    from research.portfolio.ui import planner

    return planner


def _weight_methods() -> list[str]:
    from ensemble.weight_layer import _WEIGHT_METHODS

    return sorted(_WEIGHT_METHODS)


def defaults() -> dict[str, Any]:
    """Phase options, current editable config values, option lists, output root + latest job."""

    from research.portfolio.config import PortfolioFitMode

    config = _load_config()
    planner = _planner()
    ui_defaults = planner.build_ui_defaults(config)
    initial = planner.build_ui_request(
        phase_name=ui_defaults.default_phase.value,
        tickers_text=",".join(t.name for t in ui_defaults.default_tickers),
        fallback_tickers=ui_defaults.default_tickers,
    )
    plan = planner.build_phase_plan(config, initial)
    base = planner.defaults_to_dict(ui_defaults)
    # Current editable config values + the option lists the UI form needs.
    base.update(
        {
            "tickers": [t.name for t in config.tickers],
            "fit_mode": config.portfolio_fit_mode.name,
            "fit_modes": [m.name for m in PortfolioFitMode],
            "weight_layer_method": config.weight_layer_method,
            "weight_layer_methods": _weight_methods(),
            "target_volatility": config.target_volatility,
            "max_position_pct": config.max_position_pct,
        }
    )
    return {
        "defaults": base,
        "plan": planner.plan_to_dict(plan),
        "job": _manager().latest_job(),
    }


def start(payload: dict[str, Any]) -> dict[str, Any]:
    """Launch a portfolio phase run with optional in-UI config overrides.

    tickers go through the planner request (so ensemble/ticker remapping runs); fit mode +
    scalar knobs (weight layer, target vol, max position) are applied with ``replace`` on a fresh
    config — the canonical ``research/portfolio/config.py`` is never mutated.
    """

    from dataclasses import replace

    from ensemble.weight_layer import _WEIGHT_METHODS
    from research.portfolio.config import PortfolioFitMode

    config = _load_config()
    planner = _planner()

    overrides: dict[str, Any] = {}
    fit_mode = payload.get("fit_mode")
    if fit_mode:
        if fit_mode not in PortfolioFitMode.__members__:
            raise ValueError(f"Unknown fit_mode '{fit_mode}'.")
        overrides["portfolio_fit_mode"] = PortfolioFitMode[fit_mode]
    if payload.get("target_volatility") is not None:
        overrides["target_volatility"] = float(payload["target_volatility"])
    if payload.get("max_position_pct") is not None:
        overrides["max_position_pct"] = float(payload["max_position_pct"])
    weight_method = payload.get("weight_layer_method")
    if weight_method:
        if weight_method not in _WEIGHT_METHODS:
            raise ValueError(f"Unknown weight_layer_method '{weight_method}'.")
        overrides["weight_layer_method"] = weight_method

    # Weight-layer kwargs: SR-tilt knobs + an optional hierarchy_spec structure override.
    wl_overrides = _weight_layer_kwargs_overrides(payload)
    if wl_overrides:
        overrides["weight_layer_kwargs"] = {**config.weight_layer_kwargs, **wl_overrides}

    if overrides:
        config = replace(config, **overrides)

    ui_defaults = planner.build_ui_defaults(config)
    tickers = payload.get("tickers")
    tickers_text = ",".join(tickers) if isinstance(tickers, list) else tickers
    request = planner.build_ui_request(
        phase_name=payload.get("phase") or ui_defaults.default_phase.value,
        tickers_text=tickers_text,
        fallback_tickers=ui_defaults.default_tickers,
        fit_mode_name=None,  # applied via replace above
    )
    return _manager().start_job(config, request)


def _weight_layer_kwargs_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull SR-tilt knobs + an optional hierarchy_spec override out of the run payload."""

    out: dict[str, Any] = {}
    if "sr_adjustment" in payload and payload["sr_adjustment"] is not None:
        out["sr_adjustment"] = bool(payload["sr_adjustment"])
    if payload.get("sr_tilt_max_depth") is not None:
        out["sr_tilt_max_depth"] = int(payload["sr_tilt_max_depth"])
    if payload.get("within_group_method"):
        out["within_group_method"] = str(payload["within_group_method"])
    for key in _SR_FLOAT_KEYS:
        if payload.get(key) is not None:
            out[key] = float(payload[key])
    hierarchy_spec = payload.get("hierarchy_spec")
    if hierarchy_spec is not None:
        from ensemble.weight_hierarchy import parse_hierarchy_spec

        parse_hierarchy_spec(hierarchy_spec)  # validate; raises ValueError if malformed
        out["hierarchy_spec"] = hierarchy_spec
    return out


def weight_layer() -> dict[str, Any]:
    """The active weight-layer policy: SR-tilt knobs, the hierarchy tree, and computed weights."""

    from research.portfolio.config import describe_weight_layer_policy, rebuild_weight_layer_kwargs

    config = _load_config()
    kwargs = rebuild_weight_layer_kwargs(config)
    sr = {
        "sr_adjustment": bool(kwargs.get("sr_adjustment", False)),
        "sr_avg": kwargs.get("sr_avg"),
        "sr_p_step": kwargs.get("sr_p_step"),
        "sr_min_years": kwargs.get("sr_min_years"),
        "sr_tilt_max_depth": kwargs.get("sr_tilt_max_depth"),
        "within_group_method": kwargs.get("within_group_method", "equal"),
        "fdm_max": kwargs.get("fdm_max"),
    }
    return {
        "method": config.weight_layer_method,
        "policy": describe_weight_layer_policy(kwargs),
        "sr": sr,
        "within_group_methods": ["equal", *_weight_methods()],
        "hierarchy": kwargs.get("hierarchy_spec"),
        "weights": _weights_from_csv(),
    }


def _weights_from_csv() -> dict[str, Any]:
    """Computed per-stream weights (with hierarchy path) from the last portfolio run."""

    if not _WEIGHTS_CSV.exists():
        return {"available": False, "phases": [], "rows": []}
    import math

    import pandas as pd

    frame = pd.read_csv(_WEIGHTS_CSV)

    def _num(value: Any) -> Any:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return value

    rows = [
        {
            "phase": str(r["phase"]),
            "path": str(r["cluster_id"]),  # e.g. root/equity_indices/mean_reversion_indices/<stream>
            "weight": _num(r.get("stream_weight")),
            "fdm": _num(r.get("fdm_multiplier")),
            "ticker": _num(r.get("stream_instrument_ticker")),
            "timeframe": _num(r.get("stream_signal_timeframe")),
            "stream_id": str(r.get("stream_or_model_id")),
            "mean_signal_corr": _num(r.get("mean_signal_correlation")),
        }
        for _, r in frame.iterrows()
    ]
    phases = sorted({row["phase"] for row in rows})
    return {"available": True, "phases": phases, "rows": rows}


def job(job_id: str) -> dict[str, Any] | None:
    return _manager().job_snapshot(job_id)


def artifacts() -> dict[str, Any]:
    """All files under the portfolio results root (tearsheets, holdout returns, reports)."""

    return {
        "groups": [
            {
                "label": "Portfolio results",
                "root": repo_relative(PORTFOLIO_RESULTS_ROOT),
                "files": artifacts_mod.list_dir(PORTFOLIO_RESULTS_ROOT),
            }
        ]
    }
