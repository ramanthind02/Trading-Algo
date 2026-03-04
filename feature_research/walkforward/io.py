from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path

from matplotlib.figure import Figure
import pandas as pd

from feature_research.walkforward.runner import WalkforwardRunReport


_FOLD_SCORE_BASE_COLS = [
    "fold_id",
    "param_label",
    "raw_objective",
    "oos_objective",
    "smoothed_objective",
    "rank",
    "selected_feature",
]
_FOLD_SCORE_ENHANCED_COLS = [
    "trade_frequency",
    "selected_in_top_k",
    "selected_long_bin",  # 0-based bin index from continuous model; None for rule-based
]


@dataclass(frozen=True)
class WalkforwardArtifactPaths:
    output_dir: Path
    tables_dir: Path
    tearsheets_dir: Path
    folds_csv: Path
    fold_scores_csv: Path
    selection_summary_csv: Path
    selected_params_detailed_csv: Path
    selection_params_and_regions_csv: Path
    fold_signal_metrics_csv: Path
    aggregate_ensemble_metrics_csv: Path
    oos_metrics_csv: Path
    aggregate_walkforward_metrics_csv: Path
    report_json: Path
    walkforward_stability_png: Path
    fold_timeline_png: Path
    summary_md: Path
    summary_html: Path
    tables_report_html: Path


def _validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def resolve_walkforward_output_dir(
    feature_type: str,
    module_name: str,
    root_dir: Path = Path("feature_research/shared_results"),
    output_subdir: str = "walkforward",
) -> Path:
    validated_feature_type = _validate_identifier(feature_type, "feature_type")
    validated_module_name = _validate_identifier(module_name, "module_name")
    return Path(root_dir) / validated_feature_type / validated_module_name / output_subdir


def _parse_param_label(param_label: str) -> dict[str, str]:
    parts = [part for part in str(param_label).split("|") if part]
    kv_pairs = [part.split("=", 1) for part in parts if "=" in part]
    return {f"param_{key.strip()}": value.strip() for key, value in kv_pairs if key.strip()}


def _to_scalar(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _build_selected_params_detailed(
    report: WalkforwardRunReport,
    research_context: dict[str, object],
) -> pd.DataFrame:
    selected_cols = [
        "fold_id",
        "param_label",
        "raw_objective",
        "oos_objective",
        "smoothed_objective",
        "rank",
        "trade_frequency",
        "selected_in_top_k",
        "selected_long_bin",
    ]
    available_cols = [col for col in selected_cols if col in report.fold_scores_df.columns]
    # Include rows selected as top-k (selected_in_top_k) or as single rank-1 (selected_feature)
    sel_top_k = (
        report.fold_scores_df["selected_in_top_k"].astype(bool)
        if "selected_in_top_k" in report.fold_scores_df.columns
        else False
    )
    sel_feature = report.fold_scores_df["selected_feature"].astype(bool)
    selection_mask = sel_top_k | sel_feature if isinstance(sel_top_k, pd.Series) else sel_feature
    selected_rows = report.fold_scores_df.loc[selection_mask, available_cols].copy()
    if selected_rows.empty:
        return pd.DataFrame()

    fold_window = report.folds_df[["fold_id", "train_start", "train_end", "test_start", "test_end"]]
    detailed = selected_rows.merge(fold_window, on="fold_id", how="left")
    objective_metric_name = getattr(report, "objective_metric_name", "")
    if objective_metric_name:
        detailed = detailed.assign(objective_metric_name=objective_metric_name)

    parsed_params = pd.DataFrame([_parse_param_label(label) for label in detailed["param_label"]])
    context_cols = pd.DataFrame(
        [{f"context_{key}": _to_scalar(value) for key, value in sorted(research_context.items())}]
    )
    context_frame = pd.concat([context_cols] * len(detailed), ignore_index=True)
    return pd.concat([detailed.reset_index(drop=True), parsed_params, context_frame], axis=1)


def _build_selection_params_and_regions(report: WalkforwardRunReport) -> pd.DataFrame:
    """One row per (fold_id, param_label) with fold window and region info when available."""
    sel_top_k = (
        report.fold_scores_df["selected_in_top_k"].astype(bool)
        if "selected_in_top_k" in report.fold_scores_df.columns
        else False
    )
    sel_feature = report.fold_scores_df["selected_feature"].astype(bool)
    selection_mask = sel_top_k | sel_feature if isinstance(sel_top_k, pd.Series) else sel_feature
    selected = report.fold_scores_df.loc[selection_mask].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "param_label",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "region_id",
                "region_size",
            ]
        )
    fold_window = report.folds_df[["fold_id", "train_start", "train_end", "test_start", "test_end"]]
    base_cols = ["fold_id", "param_label"]
    region_cols = [c for c in ("region_id", "region_size") if c in selected.columns]
    out = selected[base_cols + region_cols].drop_duplicates().merge(
        fold_window, on="fold_id", how="left"
    )
    for col in ("region_id", "region_size"):
        if col not in out.columns:
            out[col] = None
    return out


def _build_aggregate_ensemble_metrics(report: WalkforwardRunReport) -> pd.DataFrame:
    """Aggregate ensemble-level metrics over the full walkforward (per-fold Sharpe stats)."""
    if report.portfolio_results_df.empty or "oos_portfolio_sharpe" not in report.portfolio_results_df.columns:
        return pd.DataFrame(columns=["metric", "value"])
    valid = report.portfolio_results_df["oos_portfolio_sharpe"].dropna().astype(float)
    rows = [
        ("num_folds", int(len(report.portfolio_results_df))),
        ("mean_oos_portfolio_sharpe", float(valid.mean()) if not valid.empty else float("nan")),
        ("median_oos_portfolio_sharpe", float(valid.median()) if not valid.empty else float("nan")),
        ("min_oos_portfolio_sharpe", float(valid.min()) if not valid.empty else float("nan")),
        ("max_oos_portfolio_sharpe", float(valid.max()) if not valid.empty else float("nan")),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _build_aggregate_walkforward_metrics(report: WalkforwardRunReport) -> pd.DataFrame:
    """Metrics from concatenated walkforward OOS returns (aggregate test-period performance)."""
    agg_returns = getattr(report, "aggregate_oos_returns", None)
    if agg_returns is None or agg_returns.empty:
        return pd.DataFrame(columns=["metric", "value"])
    ret = agg_returns.dropna().astype(float)
    if ret.empty:
        return pd.DataFrame(columns=["metric", "value"])
    n = len(ret)
    total_return = float((1 + ret).prod() - 1) if (ret > -1).all() else float("nan")
    mean_ret = float(ret.mean())
    std_ret = float(ret.std(ddof=0))
    periods_per_year = report.timeframe.bars_per_year
    sharpe_ann = float("nan") if not std_ret or std_ret <= 0 else mean_ret / std_ret * (periods_per_year**0.5)
    downside = ret[ret < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
    sortino_ann = float("nan") if not downside_std or downside_std <= 0 else mean_ret / downside_std * (periods_per_year**0.5)
    cum = (1 + ret).cumprod()
    drawdown = (cum.cummax() - cum) / cum.cummax()
    max_dd = float(drawdown.max()) if not drawdown.empty else float("nan")
    calmar = float(total_return / abs(max_dd)) if max_dd == max_dd and abs(max_dd) > 1e-12 else float("nan")
    win_rate = float((ret > 0).mean())
    rows = [
        ("n_periods", n),
        ("total_return", total_return),
        ("mean_return", mean_ret),
        ("std_return", std_ret),
        ("sharpe_annualized", sharpe_ann),
        ("sortino_annualized", sortino_ann),
        ("max_drawdown", max_dd),
        ("calmar_ratio", calmar),
        ("win_rate", win_rate),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _build_oos_metrics(report: WalkforwardRunReport) -> pd.DataFrame:
    selected = report.fold_scores_df.loc[report.fold_scores_df["selected_feature"].astype(bool)].copy()
    total_folds = int(len(report.folds_df))
    if selected.empty:
        return pd.DataFrame(
            {
                "metric": ["num_folds", "num_selected_rows"],
                "value": [total_folds, 0],
            }
        )

    objective_metric_name = getattr(report, "objective_metric_name", "")
    metric_suffix = f"_{objective_metric_name}" if objective_metric_name else "_objective"
    objective_col = "oos_objective" if "oos_objective" in selected.columns else "raw_objective"
    raw = selected[objective_col].astype(float)
    smooth = selected["smoothed_objective"].astype(float)
    # Treat -inf/inf as missing for aggregate stats so summary table shows nan instead of -inf
    raw_finite = raw.replace([float("-inf"), float("inf")], float("nan"))
    smooth_finite = smooth.replace([float("-inf"), float("inf")], float("nan"))
    # Count how often each param_label was selected (in top-k) across folds
    chosen_counts = selected["param_label"].value_counts()
    most_selected_feature = str(chosen_counts.index[0]) if not chosen_counts.empty else ""
    most_selected_count = int(chosen_counts.iloc[0]) if not chosen_counts.empty else 0
    unique_selected = int(selected["param_label"].nunique())

    def _nan_safe_stats(s: pd.Series) -> tuple[float, float, float, float, float]:
        if s.empty or not s.notna().any():
            return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        valid = s.dropna()
        return (
            float(valid.mean()),
            float(valid.median()),
            float(valid.std(ddof=0)),
            float(valid.min()),
            float(valid.max()),
        )

    raw_mean, raw_median, raw_std, raw_min, raw_max = _nan_safe_stats(raw_finite)
    positive_raw_rate = float((raw_finite > 0.0).mean()) if raw_finite.notna().any() else float("nan")
    smooth_mean = _nan_safe_stats(smooth_finite)[0]

    metrics: list[tuple[str, object]] = [
        ("objective_metric_name", objective_metric_name),
        ("num_folds", total_folds),
        ("num_selected_rows", int(len(selected))),
        (f"mean_selected_raw{metric_suffix}", raw_mean),
        (f"median_selected_raw{metric_suffix}", raw_median),
        (f"std_selected_raw{metric_suffix}", raw_std),
        (f"min_selected_raw{metric_suffix}", raw_min),
        (f"max_selected_raw{metric_suffix}", raw_max),
        ("positive_raw_fold_rate", positive_raw_rate),
        (f"mean_selected_smoothed{metric_suffix}", smooth_mean),
        ("unique_selected_features", unique_selected),
        ("most_selected_feature", most_selected_feature),
        ("most_selected_feature_count", most_selected_count),
    ]
    if (
        not report.portfolio_results_df.empty
        and "oos_portfolio_sharpe" in report.portfolio_results_df.columns
    ):
        valid_portfolio_sharpes = report.portfolio_results_df["oos_portfolio_sharpe"].dropna().astype(float)
        metrics.append(
            (
                "mean_oos_portfolio_sharpe",
                float(valid_portfolio_sharpes.mean()) if not valid_portfolio_sharpes.empty else float("nan"),
            )
        )
    return pd.DataFrame(metrics, columns=["metric", "value"])


def _frame_to_markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows_"
    subset = frame.head(max_rows).copy()
    headers = [str(col) for col in subset.columns]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in subset.itertuples(index=False):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_summary_markdown(
    path: Path,
    feature_type: str,
    module_name: str,
    report: WalkforwardRunReport,
    oos_metrics_df: pd.DataFrame,
    selected_params_df: pd.DataFrame,
    aggregate_walkforward_metrics_df: pd.DataFrame,
) -> None:
    objective_metric_name = getattr(report, "objective_metric_name", "")

    portfolio_rows = report.portfolio_results_df.copy()
    portfolio_section = []
    if not portfolio_rows.empty and "oos_portfolio_sharpe" in portfolio_rows.columns:
        valid_sharpes = portfolio_rows["oos_portfolio_sharpe"].dropna().astype(float)
        mean_portfolio_sharpe = float(valid_sharpes.mean()) if not valid_sharpes.empty else float("nan")
        portfolio_section = [
            "",
            "## Portfolio Simulation (Stage 2)",
            f"- Mean OOS Portfolio Sharpe: {mean_portfolio_sharpe:.6f}",
            _frame_to_markdown_table(portfolio_rows, max_rows=50),
        ]

    agg_wf_section = []
    if not aggregate_walkforward_metrics_df.empty:
        agg_wf_section = [
            "",
            "## Aggregate Walkforward Test Performance",
            _frame_to_markdown_table(aggregate_walkforward_metrics_df, max_rows=50),
        ]
    lines = [
        f"# Walkforward Summary: {feature_type}/{module_name}",
        "",
        "## Overview",
        f"- Folds: {len(report.folds_df)}",
        f"- Scored rows: {len(report.fold_scores_df)}",
        f"- Selection rows: {len(report.selection_summary_df)}",
    ]
    if objective_metric_name:
        lines.append(f"- Objective metric: {objective_metric_name}")
    lines.extend(
        [
            "",
            "## Aggregated OOS Metrics",
            _frame_to_markdown_table(oos_metrics_df, max_rows=200),
            *agg_wf_section,
            "",
            "## Selected Parameters By Fold",
            _frame_to_markdown_table(selected_params_df, max_rows=50),
            "",
            "## Fold Timeline (Tabular)",
            _frame_to_markdown_table(report.folds_df, max_rows=50),
            *portfolio_section,
            "",
            "## Notes",
            "- **Readable tables**: open **`tables_report.html`** for all tabular data in one page (nav by section, scrollable tables).",
            "- **Tabular data** (CSVs) are under `tables/`: folds, fold_scores, selection_summary, selected_params_detailed, selection_params_and_regions, fold_signal_metrics, aggregate_ensemble_metrics, oos_metrics.",
            "- **Tearsheets** (QuantStats HTML reports) are under `tearsheets/`: aggregate walkforward ensemble plus per-fold ensemble and per-param-combo reports when portfolio simulation runs. Strategy returns cover only the union of fold test periods; **last fold test end** (see `context_last_fold_test_end` in selected_params_detailed or report.json) is the actual coverage end. Years with no test data (e.g. after the last fold) show as NaN in the tearsheet.",
            "- Use `tables/oos_metrics.csv` for aggregate metrics, `tables/selected_params_detailed.csv` for parameter-level analysis, `tables/fold_scores.csv` for full ranking data per fold.",
        ]
    )
    if not report.folds_df.empty and "test_end" in report.folds_df.columns:
        last_test_end = report.folds_df["test_end"].max()
        lines.append(
            f"- **Actual test coverage**: through {pd.Timestamp(last_test_end).date()} (last fold test_end). Config period_end may be later; re-run with data through that date to extend coverage."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_html(
    path: Path,
    feature_type: str,
    module_name: str,
    report: WalkforwardRunReport,
    oos_metrics_df: pd.DataFrame,
    selected_params_df: pd.DataFrame,
    aggregate_walkforward_metrics_df: pd.DataFrame,
) -> None:
    objective_metric_name = getattr(report, "objective_metric_name", "")

    portfolio_html_section = ""
    if not report.portfolio_results_df.empty and "oos_portfolio_sharpe" in report.portfolio_results_df.columns:
        valid_sharpes = report.portfolio_results_df["oos_portfolio_sharpe"].dropna().astype(float)
        mean_portfolio_sharpe = float(valid_sharpes.mean()) if not valid_sharpes.empty else float("nan")
        portfolio_html_section = (
            "<h2>Portfolio Simulation (Stage 2)</h2>"
            f"<p>Mean OOS Portfolio Sharpe: {mean_portfolio_sharpe:.6f}</p>"
            + report.portfolio_results_df.to_html(index=False, escape=True)
        )

    agg_wf_html = ""
    if not aggregate_walkforward_metrics_df.empty:
        agg_wf_html = "<h2>Aggregate Walkforward Test Performance</h2>" + aggregate_walkforward_metrics_df.to_html(index=False, escape=True)

    html = "".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'><title>",
            escape(f"Walkforward Summary: {feature_type}/{module_name}"),
            "</title><style>",
            "body{font-family:Arial,sans-serif;margin:24px;line-height:1.4}",
            "table{border-collapse:collapse;margin:12px 0;width:100%}",
            "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left}",
            "th{background:#f5f5f5}",
            "h1,h2{margin-top:24px}",
            "</style></head><body>",
            f"<h1>{escape(f'Walkforward Summary: {feature_type}/{module_name}')}</h1>",
            "<h2>Overview</h2>",
            "<ul>",
            f"<li>Folds: {len(report.folds_df)}</li>",
            f"<li>Scored rows: {len(report.fold_scores_df)}</li>",
            f"<li>Selection rows: {len(report.selection_summary_df)}</li>",
            (
                f"<li>Objective metric: {escape(objective_metric_name)}</li>"
                if objective_metric_name
                else ""
            ),
            "</ul>",
            "<h2>Aggregated OOS Metrics</h2>",
            oos_metrics_df.to_html(index=False, escape=True),
            agg_wf_html,
            "<h2>Selected Parameters By Fold</h2>",
            selected_params_df.to_html(index=False, escape=True),
            "<h2>Fold Timeline (Tabular)</h2>",
            report.folds_df.to_html(index=False, escape=True),
            portfolio_html_section,
            "</body></html>",
        ]
    )
    path.write_text(html, encoding="utf-8")


def _section_table(
    section_id: str,
    title: str,
    df: pd.DataFrame,
    max_display_rows: int = 2000,
) -> str:
    """Render a section with optional scroll for large tables."""
    if df.empty:
        table_html = "<p><em>No rows</em></p>"
    else:
        table_html = df.head(max_display_rows).to_html(index=False, escape=True, border=0)
        if len(df) > max_display_rows:
            table_html += f"<p><em>Showing first {max_display_rows} of {len(df)} rows. Use CSV for full data.</em></p>"
    return (
        f'<section id="{escape(section_id)}"><h2>{escape(title)}</h2>'
        f'<div class="table-wrap">{table_html}</div></section>'
    )


def _write_tables_report_html(
    path: Path,
    feature_type: str,
    module_name: str,
    report: WalkforwardRunReport,
    selected_params_df: pd.DataFrame,
    selection_params_and_regions_df: pd.DataFrame,
    fold_signal_metrics_df: pd.DataFrame,
    aggregate_ensemble_metrics_df: pd.DataFrame,
    oos_metrics_df: pd.DataFrame,
    aggregate_walkforward_metrics_df: pd.DataFrame,
) -> None:
    """Write a single HTML report with all tabular data for researcher UX."""
    objective_metric_name = getattr(report, "objective_metric_name", "")
    sections: list[tuple[str, str]] = [
        ("folds", "Folds"),
        ("fold_scores", "Fold scores"),
        ("selection_summary", "Selection summary"),
        ("selected_params_detailed", "Selected params (detailed)"),
        ("selection_params_and_regions", "Selection params and regions"),
        ("fold_signal_metrics", "Fold signal metrics"),
        ("aggregate_ensemble_metrics", "Aggregate ensemble metrics"),
        ("oos_metrics", "OOS metrics"),
        ("aggregate_walkforward_metrics", "Aggregate walkforward test performance"),
        ("portfolio_results", "Portfolio simulation"),
    ]
    nav_links = "".join(
        f'<li><a href="#{sid}">{escape(title)}</a></li>' for sid, title in sections
    )
    fold_score_cols = [c for c in _FOLD_SCORE_BASE_COLS if c in report.fold_scores_df.columns] + [
        c for c in _FOLD_SCORE_ENHANCED_COLS if c in report.fold_scores_df.columns
    ]
    fold_scores_subset = report.fold_scores_df[fold_score_cols] if fold_score_cols else report.fold_scores_df

    parts: list[str] = []
    parts.append(
        '<!doctype html><html><head><meta charset="utf-8"><title>'
        + escape(f"Walkforward tables: {feature_type}/{module_name}")
        + "</title><style>"
        "body{font-family:system-ui,-apple-system,sans-serif;margin:24px;line-height:1.5;max-width:1400px}"
        "h1{font-size:1.5rem;margin-bottom:8px}"
        "h2{font-size:1.2rem;margin-top:32px;margin-bottom:12px;padding-top:16px;border-top:1px solid #eee}"
        "nav{background:#f8f9fa;padding:12px 16px;border-radius:6px;margin-bottom:24px}"
        "nav ul{margin:0;padding-left:20px;display:flex;flex-wrap:wrap;gap:0 24px}"
        "nav a{color:#0550ae;text-decoration:none}"
        "nav a:hover{text-decoration:underline}"
        "table{border-collapse:collapse;width:100%;margin:8px 0;font-size:0.9rem}"
        "th,td{border:1px solid #ddd;padding:8px 10px;text-align:left}"
        "th{background:#f0f0f0;font-weight:600}"
        "tr:nth-child(even){background:#fafafa}"
        ".table-wrap{overflow-x:auto;max-height:70vh;overflow-y:auto;margin-bottom:16px}"
        "section:target h2{color:#0550ae}"
        "</style></head><body>"
        f"<h1>{escape(f'Walkforward tables: {feature_type} / {module_name}')}</h1>"
        "<p>All tabular outputs in one place. Use the nav to jump to a section. CSVs in <code>tables/</code> for scripted use.</p>"
        f"{f'<p>Objective metric: {escape(objective_metric_name)}</p>' if objective_metric_name else ''}"
        f"<nav><ul>{nav_links}</ul></nav>"
    )
    parts.append(
        _section_table("folds", "Folds", report.folds_df)
    )
    parts.append(
        _section_table("fold_scores", "Fold scores", fold_scores_subset)
    )
    parts.append(
        _section_table("selection_summary", "Selection summary", report.selection_summary_df)
    )
    parts.append(
        _section_table("selected_params_detailed", "Selected params (detailed)", selected_params_df)
    )
    parts.append(
        _section_table("selection_params_and_regions", "Selection params and regions", selection_params_and_regions_df)
    )
    parts.append(
        _section_table("fold_signal_metrics", "Fold signal metrics", fold_signal_metrics_df)
    )
    parts.append(
        _section_table("aggregate_ensemble_metrics", "Aggregate ensemble metrics", aggregate_ensemble_metrics_df)
    )
    parts.append(
        _section_table("oos_metrics", "OOS metrics", oos_metrics_df)
    )
    parts.append(
        _section_table("aggregate_walkforward_metrics", "Aggregate walkforward test performance", aggregate_walkforward_metrics_df)
    )
    parts.append(
        _section_table("portfolio_results", "Portfolio simulation", report.portfolio_results_df)
    )
    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def write_walkforward_artifacts(
    report: WalkforwardRunReport,
    walkforward_stability_figure: Figure,
    fold_timeline_figure: Figure,
    feature_type: str,
    module_name: str,
    root_dir: Path = Path("feature_research/shared_results"),
    research_context: dict[str, object] | None = None,
    output_subdir: str = "walkforward",
) -> WalkforwardArtifactPaths:
    validated_feature_type = _validate_identifier(feature_type, "feature_type")
    validated_module_name = _validate_identifier(module_name, "module_name")
    output_dir = resolve_walkforward_output_dir(
        feature_type=validated_feature_type,
        module_name=validated_module_name,
        root_dir=root_dir,
        output_subdir=output_subdir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    tearsheets_dir = output_dir / "tearsheets"

    paths = WalkforwardArtifactPaths(
        output_dir=output_dir,
        tables_dir=tables_dir,
        tearsheets_dir=tearsheets_dir,
        folds_csv=tables_dir / "folds.csv",
        fold_scores_csv=tables_dir / "fold_scores.csv",
        selection_summary_csv=tables_dir / "selection_summary.csv",
        selected_params_detailed_csv=tables_dir / "selected_params_detailed.csv",
        selection_params_and_regions_csv=tables_dir / "selection_params_and_regions.csv",
        fold_signal_metrics_csv=tables_dir / "fold_signal_metrics.csv",
        aggregate_ensemble_metrics_csv=tables_dir / "aggregate_ensemble_metrics.csv",
        oos_metrics_csv=tables_dir / "oos_metrics.csv",
        aggregate_walkforward_metrics_csv=tables_dir / "aggregate_walkforward_metrics.csv",
        report_json=output_dir / "report.json",
        walkforward_stability_png=output_dir / "walkforward_stability.png",
        fold_timeline_png=output_dir / "fold_timeline.png",
        summary_md=output_dir / "summary.md",
        summary_html=output_dir / "summary.html",
        tables_report_html=output_dir / "tables_report.html",
    )

    report.folds_df[
        [
            "fold_id",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "train_samples",
            "test_samples",
        ]
    ].to_csv(paths.folds_csv, index=False, lineterminator="\n")
    fold_score_cols = [col for col in _FOLD_SCORE_BASE_COLS if col in report.fold_scores_df.columns] + [
        col for col in _FOLD_SCORE_ENHANCED_COLS if col in report.fold_scores_df.columns
    ]
    report.fold_scores_df[fold_score_cols].to_csv(
        paths.fold_scores_csv, index=False, lineterminator="\n"
    )
    report.selection_summary_df[
        [
            "fold_id",
            "selected_feature",
            "selected_raw_objective",
            "selected_smoothed_objective",
            "top_k_features",
        ]
    ].to_csv(paths.selection_summary_csv, index=False, lineterminator="\n")

    context = dict(research_context or {})
    # Actual coverage: tearsheet/aggregate returns only cover fold test periods
    if not report.folds_df.empty and "test_end" in report.folds_df.columns:
        last_test_end = report.folds_df["test_end"].max()
        context["last_fold_test_end"] = str(pd.Timestamp(last_test_end).date())
    agg_ret = getattr(report, "aggregate_oos_returns", None)
    if agg_ret is not None and hasattr(agg_ret, "index") and len(agg_ret.index) > 0:
        context["aggregate_returns_last_date"] = str(pd.Timestamp(agg_ret.index.max()).date())

    selected_params_df = _build_selected_params_detailed(report=report, research_context=context)
    selected_params_df.to_csv(paths.selected_params_detailed_csv, index=False, lineterminator="\n")

    selection_params_and_regions_df = _build_selection_params_and_regions(report)
    selection_params_and_regions_df.to_csv(
        paths.selection_params_and_regions_csv, index=False, lineterminator="\n"
    )

    fold_signal_metrics_to_write = getattr(report, "fold_signal_metrics_df", None)
    if fold_signal_metrics_to_write is None or fold_signal_metrics_to_write.empty:
        fold_signal_metrics_to_write = pd.DataFrame(columns=["fold_id", "signal_name", "oos_sharpe"])
    fold_signal_metrics_to_write.to_csv(
        paths.fold_signal_metrics_csv, index=False, lineterminator="\n"
    )

    aggregate_ensemble_metrics_df = _build_aggregate_ensemble_metrics(report)
    aggregate_ensemble_metrics_df.to_csv(
        paths.aggregate_ensemble_metrics_csv, index=False, lineterminator="\n"
    )

    oos_metrics_df = _build_oos_metrics(report)
    oos_metrics_df.to_csv(paths.oos_metrics_csv, index=False, lineterminator="\n")

    aggregate_walkforward_metrics_df = _build_aggregate_walkforward_metrics(report)
    aggregate_walkforward_metrics_df.to_csv(
        paths.aggregate_walkforward_metrics_csv, index=False, lineterminator="\n"
    )

    walkforward_stability_figure.savefig(paths.walkforward_stability_png)
    fold_timeline_figure.savefig(paths.fold_timeline_png)

    _write_summary_markdown(
        path=paths.summary_md,
        feature_type=validated_feature_type,
        module_name=validated_module_name,
        report=report,
        oos_metrics_df=oos_metrics_df,
        selected_params_df=selected_params_df,
        aggregate_walkforward_metrics_df=aggregate_walkforward_metrics_df,
    )
    _write_summary_html(
        path=paths.summary_html,
        feature_type=validated_feature_type,
        module_name=validated_module_name,
        report=report,
        oos_metrics_df=oos_metrics_df,
        selected_params_df=selected_params_df,
        aggregate_walkforward_metrics_df=aggregate_walkforward_metrics_df,
    )

    _write_tables_report_html(
        path=paths.tables_report_html,
        feature_type=validated_feature_type,
        module_name=validated_module_name,
        report=report,
        selected_params_df=selected_params_df,
        selection_params_and_regions_df=selection_params_and_regions_df,
        fold_signal_metrics_df=fold_signal_metrics_to_write,
        aggregate_ensemble_metrics_df=aggregate_ensemble_metrics_df,
        oos_metrics_df=oos_metrics_df,
        aggregate_walkforward_metrics_df=aggregate_walkforward_metrics_df,
    )

    tearsheet_files: list[str] = []
    if paths.tearsheets_dir.exists():
        tearsheet_files = sorted(
            str(p.relative_to(paths.tearsheets_dir)) for p in paths.tearsheets_dir.rglob("*.html")
        )

    report_payload = {
        "feature_type": validated_feature_type,
        "module_name": validated_module_name,
        "output_dir": str(output_dir),
        "tables_dir": str(paths.tables_dir),
        "tearsheets_dir": str(paths.tearsheets_dir),
        "tearsheet_files": tearsheet_files,
        "objective_metric_name": getattr(report, "objective_metric_name", ""),
        "timeframe": report.timeframe.name,
        "artifact_files": {
            "folds_csv": str(paths.folds_csv),
            "fold_scores_csv": str(paths.fold_scores_csv),
            "selection_summary_csv": str(paths.selection_summary_csv),
            "selected_params_detailed_csv": str(paths.selected_params_detailed_csv),
            "selection_params_and_regions_csv": str(paths.selection_params_and_regions_csv),
            "fold_signal_metrics_csv": str(paths.fold_signal_metrics_csv),
            "aggregate_ensemble_metrics_csv": str(paths.aggregate_ensemble_metrics_csv),
            "oos_metrics_csv": str(paths.oos_metrics_csv),
            "aggregate_walkforward_metrics_csv": str(paths.aggregate_walkforward_metrics_csv),
            "report_json": str(paths.report_json),
            "walkforward_stability_png": str(paths.walkforward_stability_png),
            "fold_timeline_png": str(paths.fold_timeline_png),
            "summary_md": str(paths.summary_md),
            "summary_html": str(paths.summary_html),
            "tables_report_html": str(paths.tables_report_html),
        },
        "research_context": {key: _to_scalar(value) for key, value in sorted(context.items())},
        "row_counts": {
            "folds": int(len(report.folds_df)),
            "fold_scores": int(len(report.fold_scores_df)),
            "selection_summary": int(len(report.selection_summary_df)),
            "portfolio_results": int(len(report.portfolio_results_df)),
            "selected_params_detailed": int(len(selected_params_df)),
            "selection_params_and_regions": int(len(selection_params_and_regions_df)),
            "fold_signal_metrics": int(len(fold_signal_metrics_to_write)),
            "aggregate_ensemble_metrics": int(len(aggregate_ensemble_metrics_df)),
            "oos_metrics": int(len(oos_metrics_df)),
            "aggregate_walkforward_metrics": int(len(aggregate_walkforward_metrics_df)),
        },
    }
    paths.report_json.write_text(
        json.dumps(report_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )

    return paths
