"""Parse the canonical exploration CSVs into chart-ready data for the results dashboard.

Keeps the frontend dumb: the backend owns the (stable) CSV schemas and emits typed structures
for the plateau chart, equity curves, the per-combo grid table, and a headline metric panel.

Sources (written by the exploration phase):
  * ``<viz>/param_sensitivity.csv``  — per-combo sharpe / t_stat / sortino
  * ``<viz>/param_combo_long.csv``   — per-combo param values (long format)
  * ``<viz>/equity_curve.csv``       — per-combo, per-ticker cumulative return
  * ``<reports>/robustness_combinations.csv`` — per-combo NW sharpe + which is best
  * ``<reports>/robustness_summary.csv``      — feature-level headline (DSR, N_eff, t, …)

Both ``<viz>`` and ``<reports>`` are per-run: the run manager isolates each run's CSVs under
a run-id subfolder, so results read back the artifacts of that specific run (not "latest wins").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from frontend.api.paths import repo_relative as _repo_relative

# Param keys in param_combo_long that are plumbing, not swept hyper-parameters.
_SKIP_PARAM_KEYS = {"_bias_module", "bias_composite_module"}
_EQUITY_MAX_POINTS = 800


def _read_csv(path: Path):
    import pandas as pd

    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _maybe_number(value: Any) -> Any:
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return value


def build_results(reports_dir: str | None, viz_dir: str | None) -> dict[str, Any]:
    reports = Path(reports_dir) if reports_dir else None
    viz = Path(viz_dir) if viz_dir else None
    return {
        "headline": _headline(reports),
        "grid": _grid(reports, viz),
        "plateau": _plateau(reports, viz),
        "equity": _equity(viz),
    }


def build_run_results(run: dict[str, Any]) -> dict[str, Any]:
    """Results for one run, under a uniform ``lanes`` envelope the frontend consumes::

        {
          "lanes": {"default": {"headline", "grid", "plateau", "equity"}},  # build_results() shape
          "lane_order": ["default"],
          "primary_lane": "default",
        }

    Research is single-feed, so there is exactly one ``default`` lane built from the run's per-run
    ``reports_dir`` / ``viz_dir``. Validation runs dispatch to a validation-specific reader that
    reads from the walkforward output files (equity_curve_validation_only.csv + robustness JSON).
    """

    phase = run.get("phase", "exploration")
    if phase == "validation":
        viz_dir = run.get("viz_dir")
        viz = Path(viz_dir) if viz_dir else None
        lane: dict[str, Any] = {
            "headline": _validation_headline(viz),
            "grid": {"available": False, "rows": [], "param_keys": []},
            "plateau": {"available": False},
            "equity": {"available": False, "series": [], "combos": [], "tickers": []},
            # Validation-specific fields the frontend uses to build a different layout:
            "validation_summary": _validation_summary(viz),
            "validation_images": _validation_images(viz),
        }
    else:
        lane = build_results(run.get("reports_dir"), run.get("viz_dir"))

    return {
        "lanes": {"default": lane},
        "lane_order": ["default"],
        "primary_lane": "default",
    }


def primary_view_dirs(run: dict[str, Any]) -> tuple[str | None, str | None]:
    """The ``(reports_dir, viz_dir)`` a single-view consumer (``/artifacts``, conditional-returns)
    reads for ``run`` — its per-run output dirs."""

    return run.get("reports_dir"), run.get("viz_dir")


def _params_by_combo(viz: Path | None) -> dict[str, dict[str, Any]]:
    """Wide per-combo param map from param_combo_long.csv (plumbing keys dropped)."""

    if viz is None:
        return {}
    path = viz / "param_combo_long.csv"
    if not path.exists():
        return {}
    frame = _read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = str(row["param_key"])
        if key in _SKIP_PARAM_KEYS:
            continue
        label = str(row["param_combo_label"])
        out.setdefault(label, {})[key] = _maybe_number(row["param_value"])
    return out


_VALIDATION_IMAGE_LABELS: dict[str, str] = {
    "validation_sharpe_comparison.png": "IS vs Validation Sharpe",
    "validation_equity_bands.png": "Equity with Confidence Bands",
    "validation_z_cusum_rolling_sr.png": "CUSUM / Rolling Sharpe",
    "validation_rank_scatter.png": "Rank Correlation (IS vs Val)",
    "portfolio_addition_delta_sr_histogram.png": "Portfolio SR Improvement Distribution",
    "portfolio_addition_correlation_heatmap.png": "Portfolio Correlation Heatmap",
    "portfolio_addition_drawdown_correlation.png": "Drawdown Correlation",
    "portfolio_addition_risk_impact_deltas.png": "Risk Impact Deltas",
}

# Preferred display order: validation charts first, then gate charts.
_VALIDATION_IMAGE_ORDER = list(_VALIDATION_IMAGE_LABELS)


def _validation_images(viz: Path | None) -> list[dict[str, str]]:
    """Return ordered list of {label, path} for validation matplotlib PNGs."""

    if viz is None:
        return []
    img_dir = viz / "matplotlib"
    if not img_dir.is_dir():
        return []
    images: list[dict[str, str]] = []
    # Emit in preferred order first, then any extra PNGs alphabetically.
    found = {p.name: p for p in img_dir.iterdir() if p.suffix == ".png"}
    for name in _VALIDATION_IMAGE_ORDER:
        if name in found:
            images.append({"label": _VALIDATION_IMAGE_LABELS[name], "path": _repo_relative(found.pop(name))})
    for name, p in sorted(found.items()):
        images.append({"label": name.replace("_", " ").replace(".png", "").title(), "path": _repo_relative(p)})
    return images


def _validation_summary(viz: Path | None) -> dict[str, Any]:
    """Key metrics from validation_robustness_report.json and portfolio_addition_report.json."""

    if viz is None:
        return {}
    import json as _json

    result: dict[str, Any] = {}

    rob_path = viz / "validation_robustness_report.json"
    if rob_path.exists():
        data = _json.loads(rob_path.read_text(encoding="utf-8"))
        sc = data.get("sharpe_comparison", {})
        cusum = data.get("cusum", {})
        result.update({
            "sr_is": _jsonable(sc.get("sr_is")),
            "sr_val": _jsonable(sc.get("sr_val")),
            "degradation_ratio": _jsonable(sc.get("degradation_ratio")),
            "degradation_z": _jsonable(sc.get("degradation_z")),
            "ci_overlap": sc.get("ci_overlap"),
            "robustness_passed": sc.get("passed"),
            "robustness_interpretation": sc.get("interpretation"),
            "cusum_break": bool(cusum.get("break_detected", False)),
        })

    gate_path = viz / "portfolio_addition_report.json"
    if gate_path.exists():
        gate = _json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("skipped"):
            meta = gate.get("meta", {})
            context = gate.get("context", {})
            result.update({
                "gate_passed": gate.get("passed"),
                "gate_interpretation": gate.get("interpretation"),
                "gate_n_existing": meta.get("n_existing_strategies"),
                "gate_weight_method": meta.get("weight_layer_method"),
                "gate_mean_peer_corr": _jsonable(
                    sum(p["pearson_vs_candidate"] for p in context.get("pairwise_peers", []))
                    / max(len(context.get("pairwise_peers", [])), 1)
                    if context.get("pairwise_peers")
                    else None
                ),
            })

    return result


def _validation_headline(viz: Path | None) -> dict[str, Any] | None:
    """Headline for a completed validation run — reads validation_robustness_report.json."""

    summary = _validation_summary(viz)
    if not summary:
        return None

    gate_passed = summary.get("gate_passed")
    interp = summary.get("robustness_interpretation") or ""
    if gate_passed is not None:
        gate_label = "Portfolio gate: PASSED" if gate_passed else "Portfolio gate: FAILED"
        interp = f"{interp}\n{gate_label}" if interp else gate_label

    return {
        "best_param_combo_label": "validation walkforward",
        "selection_metric": "IS Sharpe → val Sharpe",
        "n_combinations": 1,
        "raw_sharpe_annualized": summary.get("sr_is"),
        "nw_adjusted_sharpe_annualized": summary.get("sr_val"),
        "cusum_break_detected": bool(summary.get("cusum_break", False)),
        "interpretation": interp or None,
    }


def _headline(reports: Path | None) -> dict[str, Any] | None:
    if reports is None:
        return None
    path = reports / "robustness_summary.csv"
    if not path.exists():
        return None
    frame = _read_csv(path)
    if frame.empty:
        return None
    row = frame.iloc[0]
    fields = [
        "best_param_combo_label",
        "selection_metric",
        "n_combinations",
        "raw_sharpe_annualized",
        "nw_adjusted_sharpe_annualized",
        "nw_t_adjusted",
        "dsr_probability",
        "n_effective",
        "mean_pairwise_corr",
        "rolling_positive_fraction",
        "cusum_break_detected",
        "interpretation",
    ]
    return {f: _jsonable(row[f]) for f in fields if f in frame.columns}


def _grid(reports: Path | None, viz: Path | None) -> dict[str, Any]:
    if viz is None:
        return {"available": False, "rows": [], "param_keys": []}
    sens_path = viz / "param_sensitivity.csv"
    if not sens_path.exists():
        return {"available": False, "rows": [], "param_keys": []}
    sens = _read_csv(sens_path)
    params = _params_by_combo(viz)

    # robustness_combinations uses a different (display) label than param_sensitivity, so match
    # on the param_combo string by param values (e.g. "ma_period_200" & "short_period_5").
    rc_rows: list[dict[str, Any]] = []
    if reports is not None and (reports / "robustness_combinations.csv").exists():
        rc = _read_csv(reports / "robustness_combinations.csv")
        for _, row in rc.iterrows():
            rc_rows.append(
                {
                    "param_combo": str(row.get("param_combo", "")),
                    "nw_sharpe": _jsonable(row.get("nw_adjusted_sharpe_annualized")),
                    "selection_score": _jsonable(row.get("selection_score")),
                    "is_best": bool(row.get("is_best", False)),
                }
            )

    def _match_rc(combo_params: dict[str, Any]) -> dict[str, Any]:
        for rc_row in rc_rows:
            if combo_params and all(
                f"{key}_{value}" in rc_row["param_combo"] for key, value in combo_params.items()
            ):
                return rc_row
        return {}

    param_keys = sorted({k for combo in params.values() for k in combo})
    rows: list[dict[str, Any]] = []
    for _, row in sens.iterrows():
        label = str(row["param_combo_label"])
        extra = _match_rc(params.get(label, {}))
        rows.append(
            {
                "label": label,
                "params": params.get(label, {}),
                "n_observations": _jsonable(row.get("n_observations")),
                "n_nonzero_signal": _jsonable(row.get("n_nonzero_signal")),
                "sharpe": _jsonable(row.get("sharpe")),
                "t_stat": _jsonable(row.get("t_stat")),
                "sortino": _jsonable(row.get("sortino")),
                "nw_sharpe": extra.get("nw_sharpe"),
                "selection_score": extra.get("selection_score"),
                "is_best": extra.get("is_best", False),
            }
        )
    return {"available": True, "rows": rows, "param_keys": param_keys}


def _plateau(reports: Path | None, viz: Path | None) -> dict[str, Any]:
    """Swept params + per-combo metrics; the frontend picks 1D bar vs 2D heatmap."""

    grid = _grid(reports, viz)
    if not grid["available"]:
        return {"available": False}
    rows = grid["rows"]
    # A param is "swept" if it takes more than one distinct value across combos.
    swept: list[str] = []
    for key in grid["param_keys"]:
        values = {str(r["params"].get(key)) for r in rows}
        if len(values) > 1:
            swept.append(key)
    return {
        "available": True,
        "swept_params": swept,
        "metric_options": ["t_stat", "sharpe", "sortino", "nw_sharpe"],
        "points": [
            {
                "label": r["label"],
                "params": r["params"],
                "metrics": {
                    "t_stat": r["t_stat"],
                    "sharpe": r["sharpe"],
                    "sortino": r["sortino"],
                    "nw_sharpe": r["nw_sharpe"],
                },
                "is_best": r["is_best"],
            }
            for r in rows
        ],
    }


def _equity(viz: Path | None) -> dict[str, Any]:
    if viz is None:
        return {"available": False, "series": [], "combos": [], "tickers": []}
    path = viz / "equity_curve.csv"
    if not path.exists():
        # Validation writes equity_curve_validation_only.csv (same schema, different filename).
        path = viz / "equity_curve_validation_only.csv"
    if not path.exists():
        return {"available": False, "series": [], "combos": [], "tickers": []}
    frame = _read_csv(path)
    combos = sorted(frame["param_combo_label"].astype(str).unique().tolist())
    tickers = sorted(frame["ticker"].astype(str).unique().tolist())
    series: list[dict[str, Any]] = []
    for (combo, ticker), group in frame.groupby(["param_combo_label", "ticker"]):
        g = group.sort_values("datetime")
        if len(g) > _EQUITY_MAX_POINTS:
            stride = max(1, len(g) // _EQUITY_MAX_POINTS)
            g = g.iloc[::stride]
        series.append(
            {
                "combo": str(combo),
                "ticker": str(ticker),
                "datetime": g["datetime"].astype(str).tolist(),
                "cumulative": [_jsonable(v) for v in g["cumulative_strategy_return"].tolist()],
            }
        )
    return {"available": True, "series": series, "combos": combos, "tickers": tickers}


def _jsonable(value: Any) -> Any:
    import math

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else value
    # numpy scalar -> python
    try:
        import numpy as np

        if isinstance(value, np.generic):
            v = value.item()
            return None if isinstance(v, float) and math.isnan(v) else v
    except ImportError:
        pass
    return str(value)
