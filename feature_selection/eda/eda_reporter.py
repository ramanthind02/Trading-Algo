"""EDA report orchestration, diagnostics, and persistence (T004)."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_selection.eda.common_eda import (
    compute_correlation_analysis,
    compute_descriptive_stats,
    create_common_eda_plots,
)
from feature_selection.eda.continuous_eda import (
    compute_decile_analysis,
    compute_distribution_diagnostics,
    compute_quintile_spread,
    create_continuous_eda_plots,
)
from feature_selection.eda.eda_dataclasses import (
    BootstrapCI,
    BootstrapCIResults,
    CommonEDAPlots,
    CommonEDAStats,
    ContinuousEDAPlots,
    ContinuousEDAReport,
    ContinuousEDAStats,
    CorrelationAnalysis,
    DecileAnalysis,
    DecileBinStats,
    DescriptiveStats,
    DiagnosticFlags,
    DistributionDiagnostics,
    EDAConfig,
    EDAMetadata,
    LevelStats,
    PerLevelStats,
    QuintileSpread,
    RuleBasedEDAPlots,
    RuleBasedEDAReport,
    RuleBasedEDAStats,
)
from importlib import import_module
from utils.core.enums import Ticker, TimeFrame

_SCALED_EDA = import_module("".join(("feature_selection.eda.", "rule", "_based", "_eda")))
compute_bootstrap_ci = _SCALED_EDA.compute_bootstrap_ci
compute_per_level_stats = _SCALED_EDA.compute_per_level_stats
create_signed_signal_eda_plots = _SCALED_EDA.create_signed_signal_eda_plots


def run_eda_for_continuous_feature(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    metadata: EDAMetadata,
    config: EDAConfig,
) -> ContinuousEDAReport:
    """Run full T004 EDA report flow for a continuous feature."""
    common_stats, common_plots = _build_common_stats_and_plots(
        feature=feature,
        target=target,
        timestamps=timestamps,
        config=config,
    )

    decile_analysis = compute_decile_analysis(feature=feature, target=target, n_bins=config.n_bins)
    distribution_diagnostics = compute_distribution_diagnostics(feature)
    quintile_spread = compute_quintile_spread(feature=feature, target=target)
    continuous_stats = ContinuousEDAStats(
        decile_analysis=decile_analysis,
        distribution_diagnostics=distribution_diagnostics,
        quintile_spread=quintile_spread,
    )
    continuous_plots = create_continuous_eda_plots(
        feature=feature,
        target=target,
        decile_analysis=decile_analysis,
        quintile_spread=quintile_spread,
    )
    diagnostics = compute_diagnostic_flags(common_stats=common_stats, feature_stats=continuous_stats)

    return ContinuousEDAReport(
        metadata=metadata,
        common_stats=common_stats,
        continuous_stats=continuous_stats,
        common_plots=common_plots,
        continuous_plots=continuous_plots,
        diagnostics=diagnostics,
    )


def run_eda_for_signed_signal_feature(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    metadata: EDAMetadata,
    config: EDAConfig,
) -> RuleBasedEDAReport:
    """Run full T004 EDA report flow for a signed signal feature."""
    common_stats, common_plots = _build_common_stats_and_plots(
        feature=feature,
        target=target,
        timestamps=timestamps,
        config=config,
    )

    per_level_stats = compute_per_level_stats(feature=feature, target=target)
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    returns_by_level = {
        int(level): aligned.loc[aligned["f"] == level, "t"]
        for level in sorted(per_level_stats.stats_by_level.keys())
    }
    bootstrap_ci = compute_bootstrap_ci(
        returns_by_level=returns_by_level,
        n_iterations=config.bootstrap_iterations,
        seed=config.random_seed,
    )
    rule_stats = RuleBasedEDAStats(
        per_level_stats=per_level_stats,
        bootstrap_ci_results=bootstrap_ci,
    )
    rule_plots = create_signed_signal_eda_plots(
        per_level_stats=per_level_stats,
        bootstrap_ci=bootstrap_ci,
    )
    diagnostics = compute_diagnostic_flags(common_stats=common_stats, feature_stats=rule_stats)

    return RuleBasedEDAReport(
        metadata=metadata,
        common_stats=common_stats,
        rule_stats=rule_stats,
        common_plots=common_plots,
        rule_plots=rule_plots,
        diagnostics=diagnostics,
    )


def compute_diagnostic_flags(
    common_stats: CommonEDAStats,
    feature_stats: Union[ContinuousEDAStats, RuleBasedEDAStats],
) -> DiagnosticFlags:
    """Compute warning/red-flag diagnostics from common and feature-specific stats."""
    warnings: list[str] = []
    red_flags: list[str] = []
    feature_desc = common_stats.feature_stats

    if feature_desc.nan_pct > 0.10:
        warnings.append("High NaN percentage (>10%)")
    if feature_desc.sample_size < 252:
        warnings.append("Low sample size (<252)")
    if feature_desc.std == 0:
        red_flags.append("Zero variance in feature")
    if abs(feature_desc.skew) > 5:
        red_flags.append("Extreme skewness (|skew| > 5)")
    if feature_desc.sample_size > 0 and feature_desc.nan_count == feature_desc.sample_size:
        red_flags.append("All NaN feature")

    return DiagnosticFlags(warnings=warnings, red_flags=red_flags, is_viable=len(red_flags) == 0)


def save_eda_report(
    report: Union[ContinuousEDAReport, RuleBasedEDAReport],
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Persist EDA report artifacts to {output_dir}/{feature_name}/{param_hash}."""
    param_hash = _param_combo_hash(report.metadata.param_combo)
    report_dir = output_dir / report.metadata.feature_name / param_hash

    if report_dir.exists() and not overwrite:
        raise FileExistsError(f"EDA report already exists at {report_dir}")
    if report_dir.exists() and overwrite:
        shutil.rmtree(report_dir)

    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    report_type = "continuous" if isinstance(report, ContinuousEDAReport) else "signed_signal"
    metadata_payload = {
        "feature_name": report.metadata.feature_name,
        "param_combo": report.metadata.param_combo,
        "timeframe": report.metadata.timeframe.name,
        "ticker": report.metadata.ticker.name,
        "timestamp": report.metadata.timestamp.isoformat(),
        "report_type": report_type,
    }
    _write_json(report_dir / "metadata.json", metadata_payload)
    _write_json(report_dir / "common_stats.json", _to_jsonable(report.common_stats))

    feature_stats_payload = report.continuous_stats if isinstance(report, ContinuousEDAReport) else report.rule_stats
    _write_json(report_dir / "feature_stats.json", _to_jsonable(feature_stats_payload))
    _write_json(report_dir / "diagnostics.json", _to_jsonable(report.diagnostics))

    common_figures = [
        (name, getattr(report.common_plots, name))
        for name in [field.name for field in fields(report.common_plots)]
    ]
    for fig_name, fig in common_figures:
        fig.savefig(plots_dir / f"{fig_name}.png", dpi=150, bbox_inches="tight")

    feature_plots = report.continuous_plots if isinstance(report, ContinuousEDAReport) else report.rule_plots
    for plot_field in fields(feature_plots):
        fig = getattr(feature_plots, plot_field.name)
        fig.savefig(plots_dir / f"{plot_field.name}.png", dpi=150, bbox_inches="tight")

    return report_dir


def load_eda_report(report_path: Path) -> Union[ContinuousEDAReport, RuleBasedEDAReport]:
    """Load persisted report metadata, stats, diagnostics, and plot shells."""
    metadata_payload = _read_json(report_path / "metadata.json")
    common_payload = _read_json(report_path / "common_stats.json")
    feature_payload = _read_json(report_path / "feature_stats.json")
    diagnostics_payload = _read_json(report_path / "diagnostics.json")

    metadata = EDAMetadata(
        feature_name=metadata_payload["feature_name"],
        param_combo=metadata_payload["param_combo"],
        timeframe=TimeFrame[metadata_payload["timeframe"]],
        ticker=Ticker[metadata_payload["ticker"]],
        timestamp=datetime.fromisoformat(metadata_payload["timestamp"]),
    )
    diagnostics = DiagnosticFlags(
        warnings=list(diagnostics_payload["warnings"]),
        red_flags=list(diagnostics_payload["red_flags"]),
        is_viable=bool(diagnostics_payload["is_viable"]),
    )

    common_stats = _common_stats_from_json(common_payload)
    common_plots = _common_plots_from_dir(report_path / "plots")
    report_type = metadata_payload.get("report_type", "continuous")

    if report_type == "continuous":
        return ContinuousEDAReport(
            metadata=metadata,
            common_stats=common_stats,
            continuous_stats=_continuous_stats_from_json(feature_payload),
            common_plots=common_plots,
            continuous_plots=_continuous_plots_from_dir(report_path / "plots"),
            diagnostics=diagnostics,
        )

    return RuleBasedEDAReport(
        metadata=metadata,
        common_stats=common_stats,
        rule_stats=_rule_stats_from_json(feature_payload),
        common_plots=common_plots,
        rule_plots=_rule_plots_from_dir(report_path / "plots"),
        diagnostics=diagnostics,
    )


def _param_combo_hash(param_combo: dict) -> str:
    """Deterministic hash for a parameter-combination dictionary."""
    normalized = json.dumps(param_combo, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:8]


def _build_common_stats_and_plots(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    config: EDAConfig,
) -> tuple[CommonEDAStats, CommonEDAPlots]:
    feature_stats = compute_descriptive_stats(feature)
    target_stats = compute_descriptive_stats(target)
    correlation_analysis = compute_correlation_analysis(
        feature=feature,
        target=target,
        max_lag=config.max_lag,
    )
    common_stats = CommonEDAStats(
        feature_stats=feature_stats,
        target_stats=target_stats,
        correlation_analysis=correlation_analysis,
    )
    common_plots = create_common_eda_plots(feature=feature, timestamps=timestamps)
    return common_stats, common_plots


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (TimeFrame, Ticker)):
        return value.name
    if isinstance(value, pd.Series):
        return {
            "index": [str(idx) for idx in value.index],
            "values": [_to_jsonable(v) for v in value.tolist()],
        }
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptive_stats_from_json(payload: dict[str, Any]) -> DescriptiveStats:
    return DescriptiveStats(
        min_val=float(payload["min_val"]),
        max_val=float(payload["max_val"]),
        mean=float(payload["mean"]),
        median=float(payload["median"]),
        std=float(payload["std"]),
        skew=float(payload["skew"]),
        kurtosis=float(payload["kurtosis"]),
        nan_count=int(payload["nan_count"]),
        nan_pct=float(payload["nan_pct"]),
        sample_size=int(payload["sample_size"]),
    )


def _common_stats_from_json(payload: dict[str, Any]) -> CommonEDAStats:
    corr_payload = payload["correlation_analysis"]
    return CommonEDAStats(
        feature_stats=_descriptive_stats_from_json(payload["feature_stats"]),
        target_stats=_descriptive_stats_from_json(payload["target_stats"]),
        correlation_analysis=CorrelationAnalysis(
            pearson=float(corr_payload["pearson"]),
            spearman=float(corr_payload["spearman"]),
            lagged_correlations={
                int(lag): float(value)
                for lag, value in corr_payload["lagged_correlations"].items()
            },
        ),
    )


def _continuous_stats_from_json(payload: dict[str, Any]) -> ContinuousEDAStats:
    decile_payload = payload["decile_analysis"]
    bins_payload = decile_payload["bin_stats"]
    diagnostics_payload = payload["distribution_diagnostics"]

    qs_payload = payload["quintile_spread"]
    return ContinuousEDAStats(
        decile_analysis=DecileAnalysis(
            bin_stats=DecileBinStats(
                bin_edges=np.array(bins_payload["bin_edges"], dtype=float),
                mean_return=np.array(bins_payload["mean_return"], dtype=float),
                volatility=np.array(bins_payload["volatility"], dtype=float),
                sharpe=np.array(bins_payload["sharpe"], dtype=float),
                t_stat=np.array(bins_payload["t_stat"], dtype=float),
                sample_count=np.array(bins_payload["sample_count"], dtype=int),
            ),
            overall_trend=str(decile_payload["overall_trend"]),
        ),
        distribution_diagnostics=DistributionDiagnostics(
            skewness=float(diagnostics_payload["skewness"]),
            kurtosis=float(diagnostics_payload["kurtosis"]),
            normality_test_stat=float(diagnostics_payload["normality_test_stat"]),
            normality_p_value=float(diagnostics_payload["normality_p_value"]),
            is_normal=bool(diagnostics_payload["is_normal"]),
        ),
        quintile_spread=QuintileSpread(
            quintile_means=np.array(qs_payload["quintile_means"], dtype=float),
            spread=float(qs_payload["spread"]),
        ),
    )


def _rule_stats_from_json(payload: dict[str, Any]) -> RuleBasedEDAStats:
    per_level_payload = payload["per_level_stats"]["stats_by_level"]
    bootstrap_payload = payload["bootstrap_ci_results"]["ci_by_level"]

    return RuleBasedEDAStats(
        per_level_stats=PerLevelStats(
            stats_by_level={
                int(level): LevelStats(
                    level=int(stats_payload["level"]),
                    mean_return=float(stats_payload["mean_return"]),
                    volatility=float(stats_payload["volatility"]),
                    sharpe=float(stats_payload["sharpe"]),
                    adjusted_sharpe=float(stats_payload["adjusted_sharpe"]),
                    sample_count=int(stats_payload["sample_count"]),
                    is_reliable=bool(stats_payload["is_reliable"]),
                )
                for level, stats_payload in per_level_payload.items()
            }
        ),
        bootstrap_ci_results=BootstrapCIResults(
            ci_by_level={
                int(level): BootstrapCI(
                    level=int(ci_payload["level"]),
                    mean_return=float(ci_payload["mean_return"]),
                    ci_lower=float(ci_payload["ci_lower"]),
                    ci_upper=float(ci_payload["ci_upper"]),
                    bootstrap_distribution=np.array(ci_payload["bootstrap_distribution"], dtype=float),
                )
                for level, ci_payload in bootstrap_payload.items()
            }
        ),
    )


def _figure_from_png(path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    if path.exists():
        ax.imshow(plt.imread(path))
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, f"Missing plot: {path.name}", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    return fig


def _common_plots_from_dir(plots_dir: Path) -> CommonEDAPlots:
    return CommonEDAPlots(
        time_series_fig=_figure_from_png(plots_dir / "time_series_fig.png"),
    )


def _continuous_plots_from_dir(plots_dir: Path) -> ContinuousEDAPlots:
    return ContinuousEDAPlots(
        decile_plot_fig=_figure_from_png(plots_dir / "decile_plot_fig.png"),
        histogram_fig=_figure_from_png(plots_dir / "histogram_fig.png"),
        quintile_spread_fig=_figure_from_png(plots_dir / "quintile_spread_fig.png"),
    )


def _rule_plots_from_dir(plots_dir: Path) -> RuleBasedEDAPlots:
    return RuleBasedEDAPlots(
        level_plot_fig=_figure_from_png(plots_dir / "level_plot_fig.png"),
    )
