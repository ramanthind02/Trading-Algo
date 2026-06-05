"""Cross-validation for global WeightLayer hierarchy + SR / inv-corr settings.

Evaluates candidate ``weight_layer_kwargs`` on train→validation (production mirror)
and on expanding-window folds within the combined train+validation IS span.

Usage (repo root, venv python):

    python -m portfolio_research.weight_layer_cv
"""
from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ensemble.sr_adjustment import annualized_sharpe
from metrics.risk.drawdown import max_drawdown
from research.portfolio.config import PortfolioResearchConfig, load_config, with_rebuilt_weight_layer
from research.portfolio.holdout.rolling_eval import _load_grouped_ensembles
from research.portfolio.pipelines.portfolio_test import (
    _evaluate_phase,
    run_portfolio_research_cache_preflight,
)
from research.portfolio.weight_layer_export import _GLOBAL_SYNTHETIC_TICKER

_TRADING_DAYS = 252.0


@dataclass(frozen=True)
class WeightLayerCvArm:
    """One weight-layer configuration arm."""

    name: str
    sr_adjustment: bool
    within_group_method: str = "equal"
    sr_tilt_max_depth: int | None = 1
    description: str = ""

    def to_weight_layer_kwargs(self, base: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        merged["sr_adjustment"] = self.sr_adjustment
        merged["within_group_method"] = self.within_group_method
        if self.sr_adjustment:
            merged["sr_tilt_max_depth"] = self.sr_tilt_max_depth
        else:
            merged.pop("sr_tilt_max_depth", None)
        return merged


@dataclass(frozen=True)
class CvFold:
    fold_id: str
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class CvMetrics:
    sharpe: float
    calmar: float
    cagr: float
    max_dd: float
    buy_hold_weight: float
    n_days: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sharpe": self.sharpe,
            "calmar": self.calmar,
            "cagr": self.cagr,
            "max_dd": self.max_dd,
            "buy_hold_weight": self.buy_hold_weight,
            "n_days": self.n_days,
        }


@dataclass
class CvRunRow:
    arm: str
    fold_id: str
    metrics: CvMetrics
    weight_layer_method: str = "hierarchy_equal"


def build_cv_arms() -> tuple[WeightLayerCvArm, ...]:
    """Candidate arms spanning SR depth × within-group policy."""
    return (
        WeightLayerCvArm(
            name="hierarchy_equal",
            sr_adjustment=False,
            within_group_method="equal",
            description="Pure equal-split hierarchy (no SR, no inv-corr).",
        ),
        WeightLayerCvArm(
            name="SR_L1_equal",
            sr_adjustment=True,
            sr_tilt_max_depth=1,
            within_group_method="equal",
            description="SR tilt at asset class only; equal within L2/L3.",
        ),
        WeightLayerCvArm(
            name="SR_L1_inv",
            sr_adjustment=True,
            sr_tilt_max_depth=1,
            within_group_method="inverse_avg_pairwise_corr",
            description="SR at L1 + inverse-corr at L2/L3 (previous default).",
        ),
        WeightLayerCvArm(
            name="SR_all_equal",
            sr_adjustment=True,
            sr_tilt_max_depth=None,
            within_group_method="equal",
            description="SR tilt at all hierarchy levels; equal within leaves.",
        ),
        WeightLayerCvArm(
            name="SR_all_inv",
            sr_adjustment=True,
            sr_tilt_max_depth=None,
            within_group_method="inverse_avg_pairwise_corr",
            description="SR at all levels + inverse-corr within every group.",
        ),
        WeightLayerCvArm(
            name="SR_L2_equal",
            sr_adjustment=True,
            sr_tilt_max_depth=2,
            within_group_method="equal",
            description="SR through style groups; equal within instrument leaves.",
        ),
        WeightLayerCvArm(
            name="SR_L2_inv",
            sr_adjustment=True,
            sr_tilt_max_depth=2,
            within_group_method="inverse_avg_pairwise_corr",
            description="SR through style groups + inv-corr at L3 instrument groups.",
        ),
    )


def build_expanding_cv_folds(
    *,
    is_start: pd.Timestamp,
    is_end: pd.Timestamp,
) -> tuple[CvFold, ...]:
    """Four expanding-window folds on the combined train+validation IS span."""
    _ = is_end
    return (
        CvFold(
            fold_id="exp_2011_2013",
            fit_start=is_start,
            fit_end=pd.Timestamp("2010-12-31"),
            test_start=pd.Timestamp("2011-01-01"),
            test_end=pd.Timestamp("2013-12-31"),
        ),
        CvFold(
            fold_id="exp_2014_2016",
            fit_start=is_start,
            fit_end=pd.Timestamp("2013-12-31"),
            test_start=pd.Timestamp("2014-01-01"),
            test_end=pd.Timestamp("2016-12-31"),
        ),
        CvFold(
            fold_id="exp_2017_2019",
            fit_start=is_start,
            fit_end=pd.Timestamp("2016-12-31"),
            test_start=pd.Timestamp("2017-01-01"),
            test_end=pd.Timestamp("2019-12-31"),
        ),
        CvFold(
            fold_id="exp_2020_2022",
            fit_start=is_start,
            fit_end=pd.Timestamp("2019-12-31"),
            test_start=pd.Timestamp("2020-01-01"),
            test_end=pd.Timestamp("2022-12-31"),
        ),
    )


def build_production_validation_fold(config: PortfolioResearchConfig) -> CvFold:
    return CvFold(
        fold_id="production_validation",
        fit_start=pd.Timestamp(config.train_window.start),
        fit_end=pd.Timestamp(config.train_window.end),
        test_start=pd.Timestamp(config.validation_window.start),
        test_end=pd.Timestamp(config.validation_window.end),
    )


def _portfolio_metrics(returns: pd.Series) -> CvMetrics:
    clean = returns.astype(float).dropna().sort_index()
    if len(clean) < 2:
        return CvMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    sharpe = annualized_sharpe(clean)
    dd = float(max_drawdown(returns=clean))
    years = max(len(clean) / _TRADING_DAYS, 1e-6)
    total_return = float((1.0 + clean).prod() - 1.0)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    calmar = float(cagr / abs(dd)) if abs(dd) > 1e-12 else 0.0
    return CvMetrics(
        sharpe=sharpe,
        calmar=calmar,
        cagr=cagr,
        max_dd=dd,
        buy_hold_weight=0.0,
        n_days=len(clean),
    )


def _buy_hold_weight_fraction(weight_layer_df: pd.DataFrame) -> float:
    if weight_layer_df.empty or "stream_or_model_id" not in weight_layer_df.columns:
        return 0.0
    global_rows = weight_layer_df[
        weight_layer_df["weight_layer_ticker"].astype(str) == _GLOBAL_SYNTHETIC_TICKER
    ]
    mask = global_rows["stream_or_model_id"].astype(str).str.contains("buy_hold", na=False)
    return float(global_rows.loc[mask, "stream_weight"].sum())


def _config_for_arm(
    base: PortfolioResearchConfig,
    arm: WeightLayerCvArm,
) -> PortfolioResearchConfig:
    wl_kwargs = arm.to_weight_layer_kwargs(dict(base.weight_layer_kwargs))
    configured = replace(
        base,
        weight_layer_method="hierarchy_equal",
        weight_layer_kwargs=wl_kwargs,
    )
    return with_rebuilt_weight_layer(configured)


def evaluate_arm_on_fold(
    config: PortfolioResearchConfig,
    arm: WeightLayerCvArm,
    fold: CvFold,
    *,
    grouped_ensembles: dict[Any, list[Any]],
    unique_timeframes: list[Any],
) -> CvRunRow:
    scoped = _config_for_arm(config, arm)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        phase_result, weight_layer_df = _evaluate_phase(
            phase_title=f"CV_{arm.name}",
            output_dir_name=f"cv_scratch/{arm.name}/{fold.fold_id}",
            fit_start=fold.fit_start,
            fit_end=fold.fit_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            config=scoped,
            grouped_ensembles=grouped_ensembles,
            unique_timeframes=unique_timeframes,
            emit_tearsheets=False,
            run_purpose="metrics_only",
            collect_strategy_returns=False,
        )
    metrics = _portfolio_metrics(phase_result.combined_strategy_returns)
    metrics = replace(metrics, buy_hold_weight=_buy_hold_weight_fraction(weight_layer_df))
    return CvRunRow(arm=arm.name, fold_id=fold.fold_id, metrics=metrics)


def summarize_cv_rows(rows: Sequence[CvRunRow]) -> pd.DataFrame:
    records = [
        {
            "arm": row.arm,
            "fold_id": row.fold_id,
            **row.metrics.to_dict(),
        }
        for row in rows
    ]
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    agg = (
        frame.groupby("arm", as_index=False)
        .agg(
            mean_sharpe=("sharpe", "mean"),
            mean_calmar=("calmar", "mean"),
            mean_cagr=("cagr", "mean"),
            mean_buy_hold_weight=("buy_hold_weight", "mean"),
            min_sharpe=("sharpe", "min"),
            min_calmar=("calmar", "min"),
            n_folds=("fold_id", "count"),
        )
    )
    prod = frame[frame["fold_id"] == "production_validation"].set_index("arm")
    if not prod.empty:
        agg = agg.merge(
            prod[["sharpe", "calmar", "buy_hold_weight"]].rename(
                columns={
                    "sharpe": "prod_sharpe",
                    "calmar": "prod_calmar",
                    "buy_hold_weight": "prod_buy_hold_weight",
                }
            ),
            left_on="arm",
            right_index=True,
            how="left",
        )
    # Production validation (train→val mirror) is the primary selector; fold means
    # guard against overfitting a single window.
    agg["score"] = (
        0.25 * agg["mean_sharpe"]
        + 0.15 * agg["mean_calmar"]
        + 0.45 * agg["prod_sharpe"].fillna(agg["mean_sharpe"])
        + 0.15 * agg["prod_calmar"].fillna(agg["mean_calmar"])
    )
    return agg.sort_values("score", ascending=False).reset_index(drop=True)


def select_best_arm(summary: pd.DataFrame) -> str:
    if summary.empty:
        raise ValueError("CV summary is empty")
    return str(summary.iloc[0]["arm"])


def run_weight_layer_cv(
    config: PortfolioResearchConfig | None = None,
    *,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run CV arms; return (detail rows, summary, winning arm name)."""
    base = config if config is not None else load_config()
    out = output_dir or (Path(base.output_root) / "weight_layer_cv")
    out.mkdir(parents=True, exist_ok=True)

    run_portfolio_research_cache_preflight(base)
    grouped = _load_grouped_ensembles(base)
    unique_timeframes = sorted(grouped.keys())
    arms = build_cv_arms()

    is_start = pd.Timestamp(base.train_window.start)
    is_end = pd.Timestamp(base.validation_window.end)
    folds = (build_production_validation_fold(base), *build_expanding_cv_folds(
        is_start=is_start,
        is_end=is_end,
    ))

    rows: list[CvRunRow] = []
    for arm in arms:
        print(f"\n=== Arm: {arm.name} — {arm.description} ===")
        for fold in folds:
            print(
                f"  Fold {fold.fold_id}: fit {fold.fit_start.date()}→{fold.fit_end.date()} "
                f"| score {fold.test_start.date()}→{fold.test_end.date()}"
            )
            row = evaluate_arm_on_fold(
                base,
                arm,
                fold,
                grouped_ensembles=grouped,
                unique_timeframes=unique_timeframes,
            )
            rows.append(row)
            print(
                f"    SR={row.metrics.sharpe:.3f} Calmar={row.metrics.calmar:.3f} "
                f"buy_hold={row.metrics.buy_hold_weight:.1%}"
            )

    detail = pd.DataFrame(
        [
            {"arm": r.arm, "fold_id": r.fold_id, **r.metrics.to_dict()}
            for r in rows
        ]
    )
    summary = summarize_cv_rows(rows)
    winner = select_best_arm(summary)

    detail_path = out / "weight_layer_cv_detail.csv"
    summary_path = out / "weight_layer_cv_summary.csv"
    report_path = out / "weight_layer_cv_report.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(
        json.dumps(
            {
                "winner": winner,
                "arms": [arm.name for arm in arms],
                "folds": [
                    {
                        "fold_id": f.fold_id,
                        "fit_start": f.fit_start.date().isoformat(),
                        "fit_end": f.fit_end.date().isoformat(),
                        "test_start": f.test_start.date().isoformat(),
                        "test_end": f.test_end.date().isoformat(),
                    }
                    for f in folds
                ],
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWinner: {winner}")
    print(f"Detail: {detail_path}")
    print(f"Summary: {summary_path}")
    return detail, summary, winner


def main() -> None:
    run_weight_layer_cv()


if __name__ == "__main__":
    main()
