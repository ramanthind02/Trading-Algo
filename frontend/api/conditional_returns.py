"""Conditional-returns endpoint: bucket a run's best combo's returns by a chosen indicator.

Thin adapter over :mod:`research.feature.binning.conditional_returns`. It resolves the run's spec
+ best-performing combo, builds the analysis request from the UI's indicator / bin / regime
choices, runs the (cache-backed) extraction + bucketing, and returns chart-ready JSON.

The strategy whose returns are sliced is the run's **best combo** (``is_best`` in the grid); the
analysed window follows the run phase (exploration → train+validation, validation → test).
"""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime
from typing import Any

from frontend.api import results, spec_store


def build_conditional_returns(run: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Run the conditional-returns analysis for ``run`` with the UI's ``body`` options."""

    from research.feature.binning.conditional_returns import (
        ConditionalReturnsRequest,
        IndicatorSpec,
        RegimeSpec,
        analyze_conditional_returns,
    )
    from research.spec.adapter import resolve_windows
    from research.spec.serialization import spec_from_dict

    spec_id = run.get("spec_id")
    spec_payload = spec_store.get_spec(spec_id)["spec"]  # raises FileNotFoundError → 404
    spec = spec_from_dict(spec_payload)

    best_params = _best_combo_params(run, spec)
    start, end = _phase_window(run.get("phase", "exploration"), resolve_windows(spec))

    condition = _indicator_from(body.get("indicator"))
    if condition is None:
        raise ValueError("Provide an 'indicator' ({module, params}) to bin returns by.")

    regime_body = body.get("regime")
    regime = (
        RegimeSpec(
            indicator=_require_indicator(regime_body),
            threshold=float(regime_body.get("threshold", 0.0)),
            above=bool(regime_body.get("above", True)),
        )
        if regime_body
        else None
    )

    req = ConditionalReturnsRequest(
        tickers=tuple(spec.tickers),
        timeframe=spec.timeframe,
        start=start,
        end=end,
        data_feed=_analysis_feed(run.get("phase", "exploration"), spec),
        direction=spec.direction,
        strategy=IndicatorSpec(module_name=spec.signal.module_name, params=best_params),
        condition=condition,
        bin_mode="fixed" if str(body.get("bin_mode")) == "fixed" else "quantile",
        n_bins=int(body.get("n_bins", 5)),
        edges=tuple(float(e) for e in body.get("edges", []) if _is_number(e)),
        regime=regime,
        per_ticker=bool(body.get("per_ticker", False)),
    )

    result = analyze_conditional_returns(req)
    return _shape(result, spec, best_params, req)


# ---------------------------------------------------------------------------
# Best combo + window resolution
# ---------------------------------------------------------------------------


def _best_combo_params(run: dict[str, Any], spec: Any) -> dict[str, Any]:
    """The best combo's node params (grid ``is_best``; fallbacks: top metric, then grid center)."""

    from research.feature.binning.transforms import inflate_params_from_combo_long_table

    reports_dir, viz_dir = results.primary_view_dirs(run)
    grid = results.build_results(reports_dir, viz_dir).get("grid", {})
    rows = grid.get("rows", []) if grid.get("available") else []
    chosen: dict[str, Any] | None = None
    if rows:
        best = next((r for r in rows if r.get("is_best")), None)
        if best is None:
            best = max(rows, key=lambda r: _rank_key(r))
        chosen = dict(best.get("params") or {})
    if not chosen:
        # Single-combo or grid not yet written: fall back to the grid center the adapter uses.
        from research.spec.adapter import build_eval_bias_spec

        chosen = dict(build_eval_bias_spec(spec).get("params", {}))
    return inflate_params_from_combo_long_table(chosen)


def _rank_key(row: dict[str, Any]) -> float:
    for key in ("nw_sharpe", "selection_score", "t_stat", "sharpe"):
        value = row.get(key)
        if _is_number(value):
            return float(value)
    return float("-inf")


def _phase_window(phase: str, windows: Any) -> tuple[datetime, datetime]:
    if phase == "validation":
        return windows.test[0], windows.test[1]
    return windows.train[0], windows.validation[1]


def _analysis_feed(phase: str, spec: Any) -> str:
    """The research feed the conditional-returns analysis loads from — matching the run executor.

    The validation/test phase uses post-2018 CFD (realistic); exploration uses
    :func:`research.spec.adapter.exploration_feed_for` (DAILY futures-backed → faithful ratio
    futures; forex/intraday → CFD). ``spec.data_feed`` is an ignored back-compat label, NOT here.
    """

    from research.spec.adapter import exploration_feed_for

    if phase == "validation":
        return "cfd"
    return exploration_feed_for(spec)


# ---------------------------------------------------------------------------
# Body parsing + JSON shaping
# ---------------------------------------------------------------------------


def _indicator_from(payload: Any):
    from research.feature.binning.conditional_returns import IndicatorSpec

    if not isinstance(payload, dict):
        return None
    module = payload.get("module")
    if not module:
        return None
    return IndicatorSpec(module_name=str(module), params=_coerce_params(payload.get("params")))


def _require_indicator(payload: Any):
    indicator = _indicator_from(payload)
    if indicator is None:
        raise ValueError("Regime filter requires {module, params, threshold}.")
    return indicator


def _coerce_params(params: Any) -> dict[str, Any]:
    """Numeric-looking values → int/float (node constructors want ints, not '14')."""

    if not isinstance(params, dict):
        return {}
    return {str(k): _coerce_scalar(v) for k, v in params.items()}


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return value
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return int(f) if f == int(f) else f


def _shape(result: Any, spec: Any, best_params: dict[str, Any], req: Any) -> dict[str, Any]:
    return {
        "available": result.available,
        "reason": result.reason,
        "bin_mode": result.bin_mode,
        "n_bins": result.n_bins,
        "per_ticker": req.per_ticker,
        "tickers": result.tickers,
        "timeframe": spec.timeframe.name,
        "strategy": {
            "module": spec.signal.module_name,
            "params": {k: _jsonable(v) for k, v in best_params.items()},
            "label": result.strategy_label,
            "direction": spec.direction.value if hasattr(spec.direction, "value") else str(spec.direction),
        },
        "condition": {"module": req.condition.module_name, "label": result.condition_label},
        "regime": {"label": result.regime_label} if result.regime_label else None,
        "metric_options": ["mean_return", "sharpe", "t_stat", "hit_rate", "sortino", "cumulative"],
        "bins": [_bin_to_json(b) for b in result.bins],
    }


def _bin_to_json(bin_stat: Any) -> dict[str, Any]:
    return {k: _jsonable(v) for k, v in asdict(bin_stat).items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
