"""Weight Layer report exporter.

Persists WeightLayer diagnostics to CSV and JSON after each pipeline phase.

Outputs (written to ``output_dir/``):
- ``weights_by_group.csv``    per-(ticker × model) flat table with group weights
- ``signal_cross_ticker.csv`` models that appear in multiple tickers
- ``summary.csv``             per-ticker aggregate stats (FDM, n_models, n_groups)
- ``diagnostics.json``        full raw diagnostics dict for programmatic use
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ensemble.weight_layer import BaseWeightLayer

logger = logging.getLogger(__name__)


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert diagnostics to JSON-serializable form (str keys, no Enum/numpy)."""
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if hasattr(obj, "name"):
        return getattr(obj, "name", str(obj))
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def _family_id(model_name: str) -> str:
    """Extract feature-family group ID from a model name.

    Mirrors the logic in ``_extract_group_assignments`` (feature_family path):
    - If the name contains "::", use the left part
    - Return the first "_"-delimited token of that left part
    """
    left = model_name.split("::")[0].strip() if "::" in model_name else model_name
    return left.split("_")[0] if "_" in left else left


def _build_weights_by_group(
    diagnostics: dict,
    phase: str,
) -> pd.DataFrame:
    """Build the per-(ticker × model) weights table."""
    rows = []
    for ticker, info in diagnostics.get("tickers", {}).items():
        weights: dict = info.get("weights") or {}
        fdm = info.get("fdm", 1.0)
        mean_corr = info.get("mean_forecast_correlation", float("nan"))

        # Compute group weight = sum of model weights within the same (ticker, family)
        family_of = {model: _family_id(model) for model in weights}
        group_totals: dict[str, float] = {}
        for model, w in weights.items():
            grp = family_of[model]
            group_totals[grp] = group_totals.get(grp, 0.0) + w

        for model, model_weight in sorted(weights.items()):
            grp = family_of[model]
            rows.append({
                "phase": phase,
                "ticker": ticker,
                "feature_family": grp,
                "model_name": model,
                "model_weight": round(model_weight, 6),
                "group_weight": round(group_totals[grp], 6),
                "fdm": round(fdm, 4),
                "mean_forecast_correlation": round(mean_corr, 4),
            })

    return pd.DataFrame(rows)


def _build_cross_ticker(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-model cross-ticker presence and weight comparison table."""
    if weights_df.empty:
        return pd.DataFrame()

    all_tickers = sorted(weights_df["ticker"].unique())

    rows = []
    for model_name, grp in weights_df[["model_name", "feature_family"]].drop_duplicates().values:
        model_rows = weights_df[weights_df["model_name"] == model_name]
        present_tickers = sorted(str(t) for t in model_rows["ticker"].tolist())
        row: dict = {
            "model_name": model_name,
            "feature_family": grp,
            "tickers_present": ",".join(present_tickers),
            "n_tickers": len(present_tickers),
        }
        for t in all_tickers:
            match = model_rows[model_rows["ticker"] == t]
            row[f"weight_{t}"] = round(float(match["model_weight"].iloc[0]), 6) if not match.empty else None
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["n_tickers", "feature_family", "model_name"], ascending=[False, True, True])
    return df.reset_index(drop=True)


def _build_summary(diagnostics: dict, phase: str) -> pd.DataFrame:
    """Build per-ticker aggregate summary."""
    rows = []
    for ticker, info in diagnostics.get("tickers", {}).items():
        weights: dict = info.get("weights") or {}
        families = {_family_id(m) for m in weights}
        rows.append({
            "phase": phase,
            "ticker": ticker,
            "n_models": info.get("n_models", len(weights)),
            "n_groups": len(families),
            "fdm": round(info.get("fdm", 1.0), 4),
            "mean_forecast_correlation": round(info.get("mean_forecast_correlation", float("nan")), 4),
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def export_weight_layer_report(
    weight_layer: "BaseWeightLayer",
    phase_name: str,
    output_dir: Path,
) -> None:
    """Export weight layer diagnostics to ``output_dir``.

    Parameters
    ----------
    weight_layer:
        A fitted ``BaseWeightLayer`` instance (from ``portfolio.weight_layer``).
    phase_name:
        Label for the pipeline phase, e.g. ``"train"``, ``"validation"``, ``"test"``.
    output_dir:
        Directory where report files are written (created if absent).
    """
    if not weight_layer.is_fitted_:
        logger.warning("WeightLayer is not fitted — skipping report for phase '%s'", phase_name)
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = weight_layer.get_diagnostics()

    # --- diagnostics.json ---
    diag_path = output_dir / "diagnostics.json"
    with diag_path.open("w") as f:
        json.dump(_to_json_serializable(diagnostics), f, indent=2)
    logger.info("Weight layer diagnostics written to %s", diag_path)

    # --- weights_by_group.csv ---
    weights_df = _build_weights_by_group(diagnostics, phase_name)
    if not weights_df.empty:
        wbg_path = output_dir / "weights_by_group.csv"
        weights_df.to_csv(wbg_path, index=False)
        logger.info("Weights by group written to %s (%d rows)", wbg_path, len(weights_df))

        # --- signal_cross_ticker.csv ---
        cross_df = _build_cross_ticker(weights_df)
        if not cross_df.empty:
            ct_path = output_dir / "signal_cross_ticker.csv"
            cross_df.to_csv(ct_path, index=False)
            logger.info("Cross-ticker signal table written to %s (%d rows)", ct_path, len(cross_df))

    # --- summary.csv ---
    summary_df = _build_summary(diagnostics, phase_name)
    if not summary_df.empty:
        sum_path = output_dir / "summary.csv"
        summary_df.to_csv(sum_path, index=False)
        logger.info("Weight layer summary written to %s", sum_path)
