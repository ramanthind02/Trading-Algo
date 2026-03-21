from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime
import html
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from prop_firms.base.statistics import compute_portfolio_statistics
from prop_firms.base.portfolio_models import (
    PortfolioBatchStatistics,
    PortfolioSimulationResult,
)


@dataclass(frozen=True)
class PropFirmReportArtifacts:
    """Paths written by the portfolio report generator."""

    report_markdown_path: Path
    report_html_path: Path
    daily_timeline_csv_path: Path | None = None
    monthly_summary_csv_path: Path | None = None
    yearly_summary_csv_path: Path | None = None
    account_summaries_csv_path: Path | None = None
    events_csv_path: Path | None = None


def generate_portfolio_report(
    result: PortfolioSimulationResult,
    output_dir: Path,
    report_stem: str,
    save_csvs: bool = True,
) -> PropFirmReportArtifacts:
    """Write readable Markdown/HTML reports and optional CSV artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_portfolio_statistics([result])
    monthly_summary = _summarize_period(
        timeline=result.daily_timeline,
        period="M",
        label="month",
    )
    yearly_summary = _summarize_period(
        timeline=result.daily_timeline,
        period="Y",
        label="year",
    )
    report_markdown_path = output_dir / f"{report_stem}.md"
    report_html_path = output_dir / f"{report_stem}.html"
    report_markdown_path.write_text(
        _build_markdown_report(
            result=result,
            stats=stats,
            monthly_summary=monthly_summary,
            yearly_summary=yearly_summary,
        ),
        encoding="utf-8",
    )
    report_html_path.write_text(
        _build_html_report(
            result=result,
            stats=stats,
            monthly_summary=monthly_summary,
            yearly_summary=yearly_summary,
            report_stem=report_stem,
        ),
        encoding="utf-8",
    )

    if not save_csvs:
        return PropFirmReportArtifacts(
            report_markdown_path=report_markdown_path,
            report_html_path=report_html_path,
        )

    daily_timeline_csv_path = output_dir / f"{report_stem}_daily_timeline.csv"
    monthly_summary_csv_path = output_dir / f"{report_stem}_monthly_summary.csv"
    yearly_summary_csv_path = output_dir / f"{report_stem}_yearly_summary.csv"
    account_summaries_csv_path = output_dir / f"{report_stem}_account_summaries.csv"
    events_csv_path = output_dir / f"{report_stem}_events.csv"

    result.daily_timeline.to_csv(daily_timeline_csv_path, index=False)
    monthly_summary.to_csv(monthly_summary_csv_path, index=False)
    yearly_summary.to_csv(yearly_summary_csv_path, index=False)
    result.account_summaries.to_csv(account_summaries_csv_path, index=False)
    result.events.to_csv(events_csv_path, index=False)

    return PropFirmReportArtifacts(
        report_markdown_path=report_markdown_path,
        report_html_path=report_html_path,
        daily_timeline_csv_path=daily_timeline_csv_path,
        monthly_summary_csv_path=monthly_summary_csv_path,
        yearly_summary_csv_path=yearly_summary_csv_path,
        account_summaries_csv_path=account_summaries_csv_path,
        events_csv_path=events_csv_path,
    )


def _summarize_period(
    timeline: pd.DataFrame,
    period: str,
    label: str,
) -> pd.DataFrame:
    if timeline.empty:
        return pd.DataFrame()

    period_index = pd.to_datetime(timeline["trading_day"]).dt.to_period(period)
    grouped = timeline.assign(_period=period_index).groupby("_period", sort=True)
    summary = grouped.agg(
        net_cashflow=("net_cashflow", "sum"),
        gross_payouts=("gross_payouts", "sum"),
        trader_payouts=("trader_payouts", "sum"),
        challenge_costs=("challenge_costs", "sum"),
        activation_costs=("activation_costs", "sum"),
        purchased_challenges=("purchased_challenges", "sum"),
        challenge_passes=("challenge_passes", "sum"),
        challenge_failures=("challenge_failures", "sum"),
        funded_closures=("funded_closures", "sum"),
        avg_active_funded=("active_funded", "mean"),
        avg_active_challenges=("active_challenges", "mean"),
    ).reset_index()
    summary[label] = summary["_period"].astype(str)
    return summary.drop(columns="_period")


def _build_markdown_report(
    result: PortfolioSimulationResult,
    stats: PortfolioBatchStatistics,
    monthly_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
) -> str:
    summary = result.summary
    simulation_cfg = result.account_definition.account_code
    provider_label = _provider_report_label(result)
    event_counts = _format_event_counts(result.events)
    account_outcomes = _format_account_outcomes(result.account_summaries)
    funded_closure_reasons = _format_funded_closure_reasons(result.account_summaries)
    monthly_table = _format_table(monthly_summary.tail(12))
    yearly_table = _format_table(yearly_summary)

    return "\n".join(
        [
            f"# {provider_label} Portfolio Report: {simulation_cfg}",
            "",
            "## Summary",
            f"- Net cashflow: `${summary.net_cashflow:,.2f}`",
            f"- Trader payouts: `${summary.total_trader_payouts:,.2f}`",
            f"- Gross payouts: `${summary.total_gross_payouts:,.2f}`",
            f"- Challenge costs: `${summary.total_challenge_costs:,.2f}`",
            f"- Activation costs: `${summary.total_activation_costs:,.2f}`",
            f"- Funded accounts created: `{summary.funded_accounts_created}`",
            f"- Funded accounts closed: `{summary.funded_accounts_closed}`",
            (
                "- Funded closed from drawdown: "
                f"`{summary.funded_closed_from_drawdown}`"
            ),
            (
                "- Funded closed from max payouts: "
                f"`{summary.funded_closed_from_max_payouts}`"
            ),
            f"- Challenges purchased: `{summary.challenges_purchased}`",
            f"- Challenges failed: `{summary.challenges_failed}`",
            f"- Average active funded accounts: `{summary.average_active_funded_accounts:.2f}`",
            f"- Average active challenges: `{summary.average_active_challenges:.2f}`",
            f"- Payouts per funded account: `{summary.payouts_per_funded_account:.2f}`",
            f"- First payout day: `{summary.first_payout_day}`",
            f"- Days to first payout: `{summary.days_to_first_payout}`",
            "",
            "## EV Statistics",
            f"- Expected net cashflow: `${stats.expected_net_cashflow:,.2f}`",
            f"- Expected trader payouts: `${stats.expected_total_trader_payouts:,.2f}`",
            f"- Expected gross payouts: `${stats.expected_total_gross_payouts:,.2f}`",
            f"- Probability of negative net cashflow: `{stats.probability_negative_net_cashflow:.2%}`",
            "",
            "## Event Counts",
            event_counts,
            "",
            "## Account Outcomes",
            account_outcomes,
            "",
            "## Funded Closure Reasons",
            funded_closure_reasons,
            "",
            "## Yearly Cashflow",
            yearly_table,
            "",
            "## Monthly Cashflow (Last 12 Periods)",
            monthly_table,
            "",
            "## Output Tables",
            "- Daily timeline CSV",
            "- Monthly summary CSV",
            "- Yearly summary CSV",
            "- Account summaries CSV",
            "- Events CSV",
            "",
        ]
    )


def _build_html_report(
    result: PortfolioSimulationResult,
    stats: PortfolioBatchStatistics,
    monthly_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    report_stem: str,
) -> str:
    summary = result.summary
    provider_label = _provider_report_label(result)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    date_range = _format_date_range(result.daily_timeline)
    event_counts = _build_count_frame(
        frame=result.events,
        column_name="event_type",
    )
    account_outcomes = _build_count_frame(
        frame=result.account_summaries,
        column_name="lifecycle_state",
    )
    funded_closure_reasons = _build_funded_closure_frame(result.account_summaries)
    recent_events = (
        result.events.sort_values("trading_day", kind="stable")
        .tail(25)
        .reset_index(drop=True)
    )
    recent_accounts = (
        result.account_summaries.sort_values("start_day", kind="stable")
        .tail(25)
        .reset_index(drop=True)
    )
    csv_links = _build_csv_links(report_stem=report_stem)
    chart_cards = _build_chart_cards(
        timeline=result.daily_timeline,
        monthly_summary=monthly_summary,
    )

    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\" />",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            f"<title>{html.escape(f'{provider_label} Portfolio Report: {summary.account_code}')}</title>",
            "<style>",
            _html_styles(),
            "</style>",
            "</head>",
            "<body>",
            "<div class=\"page\">",
            "<header class=\"hero\">",
            f"<h1>{html.escape(provider_label)} Portfolio Report: {html.escape(summary.account_code)}</h1>",
            "<div class=\"hero-meta\">",
            f"<span>Generated: {html.escape(generated_at)}</span>",
            f"<span>Date range: {html.escape(date_range)}</span>",
            f"<span>Report stem: {html.escape(report_stem)}</span>",
            "</div>",
            "</header>",
            "<section>",
            "<h2>Summary</h2>",
            "<div class=\"card-grid\">",
            _metric_card("Net cashflow", _format_currency(summary.net_cashflow)),
            _metric_card("Trader payouts", _format_currency(summary.total_trader_payouts)),
            _metric_card("Gross payouts", _format_currency(summary.total_gross_payouts)),
            _metric_card("Funded created", str(summary.funded_accounts_created)),
            _metric_card("Funded closed", str(summary.funded_accounts_closed)),
            _metric_card(
                "Closed from drawdown",
                str(summary.funded_closed_from_drawdown),
            ),
            _metric_card(
                "Closed from max payouts",
                str(summary.funded_closed_from_max_payouts),
            ),
            _metric_card("Challenges purchased", str(summary.challenges_purchased)),
            _metric_card("Challenges failed", str(summary.challenges_failed)),
            _metric_card("First payout day", _format_optional_value(summary.first_payout_day)),
            _metric_card("Days to first payout", _format_optional_value(summary.days_to_first_payout)),
            "</div>",
            "</section>",
            "<section>",
            "<h2>EV Statistics</h2>",
            "<div class=\"card-grid compact\">",
            _metric_card("Expected net cashflow", _format_currency(stats.expected_net_cashflow)),
            _metric_card("Expected trader payouts", _format_currency(stats.expected_total_trader_payouts)),
            _metric_card("Expected gross payouts", _format_currency(stats.expected_total_gross_payouts)),
            _metric_card(
                "Probability negative net cashflow",
                f"{stats.probability_negative_net_cashflow:.2%}",
            ),
            _metric_card(
                "Avg active funded",
                f"{summary.average_active_funded_accounts:.2f}",
            ),
            _metric_card(
                "Avg active challenges",
                f"{summary.average_active_challenges:.2f}",
            ),
            "</div>",
            "</section>",
            "<section>",
            "<h2>Charts</h2>",
            "<div class=\"chart-grid\">",
            chart_cards,
            "</div>",
            "</section>",
            "<section class=\"two-col\">",
            "<div>",
            "<h2>Event Counts</h2>",
            _render_html_table(event_counts),
            "</div>",
            "<div>",
            "<h2>Account Outcomes</h2>",
            _render_html_table(account_outcomes),
            "</div>",
            "</section>",
            "<section>",
            "<h2>Funded Closure Reasons</h2>",
            _render_html_table(funded_closure_reasons),
            "</section>",
            "<section>",
            "<h2>Yearly Cashflow</h2>",
            _render_html_table(yearly_summary),
            "</section>",
            "<section>",
            "<h2>Monthly Cashflow (Last 12 Periods)</h2>",
            _render_html_table(monthly_summary.tail(12).reset_index(drop=True)),
            "</section>",
            "<section class=\"two-col\">",
            "<div>",
            "<h2>Recent Event Log</h2>",
            _render_html_table(recent_events),
            "</div>",
            "<div>",
            "<h2>Recent Account Summaries</h2>",
            _render_html_table(recent_accounts),
            "</div>",
            "</section>",
            "<section>",
            "<h2>Companion CSV Files</h2>",
            "<ul class=\"file-list\">",
            csv_links,
            "</ul>",
            "</section>",
            "</div>",
            "</body>",
            "</html>",
        ]
    )


def _provider_report_label(result: PortfolioSimulationResult) -> str:
    return result.account_definition.provider_name.replace("_", " ").title()


def _format_event_counts(events: pd.DataFrame) -> str:
    if events.empty:
        return "- No events"
    counts = (
        events["event_type"]
        .value_counts()
        .rename_axis("event_type")
        .reset_index(name="count")
    )
    lines = [f"- `{row.event_type}`: `{row.count}`" for row in counts.itertuples()]
    return "\n".join(lines)


def _format_account_outcomes(account_summaries: pd.DataFrame) -> str:
    if account_summaries.empty:
        return "- No accounts"

    outcome_counts = (
        account_summaries["lifecycle_state"]
        .value_counts()
        .rename_axis("lifecycle_state")
        .reset_index(name="count")
    )
    lines = [
        f"- `{row.lifecycle_state}`: `{row.count}`"
        for row in outcome_counts.itertuples()
    ]
    return "\n".join(lines)


def _format_funded_closure_reasons(account_summaries: pd.DataFrame) -> str:
    closure_frame = _build_funded_closure_frame(account_summaries)
    if closure_frame.empty:
        return "- No funded closures"
    lines = [
        f"- `{row.closure_reason}`: `{row.count}`"
        for row in closure_frame.itertuples()
    ]
    return "\n".join(lines)


def _format_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows_"
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _build_count_frame(frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[column_name, "count"])
    return (
        frame[column_name]
        .value_counts()
        .rename_axis(column_name)
        .reset_index(name="count")
    )


def _build_funded_closure_frame(account_summaries: pd.DataFrame) -> pd.DataFrame:
    if account_summaries.empty:
        return pd.DataFrame(columns=["closure_reason", "count"])
    funded_closed = account_summaries[
        account_summaries["lifecycle_state"] == "funded_closed"
    ].copy()
    if funded_closed.empty:
        return pd.DataFrame(columns=["closure_reason", "count"])
    funded_closed["closure_reason"] = funded_closed["breach_reason"].map(
        {
            "max_loss_limit": "drawdown_limit",
            "payout_limit": "max_payouts",
        }
    ).fillna("other")
    return (
        funded_closed["closure_reason"]
        .value_counts()
        .rename_axis("closure_reason")
        .reset_index(name="count")
    )


def _render_html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class=\"empty-state\">No rows</p>"
    safe_frame = frame.copy()
    safe_frame.columns = [str(column) for column in safe_frame.columns]
    return (
        "<div class=\"table-wrap\">"
        + safe_frame.to_html(
            index=False,
            border=0,
            classes="report-table",
            justify="left",
            escape=True,
        )
        + "</div>"
    )


def _metric_card(title: str, value: str) -> str:
    return (
        "<div class=\"metric-card\">"
        f"<div class=\"metric-title\">{html.escape(title)}</div>"
        f"<div class=\"metric-value\">{html.escape(value)}</div>"
        "</div>"
    )


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _format_optional_value(value: object) -> str:
    return "N/A" if value is None else str(value)


def _format_date_range(timeline: pd.DataFrame) -> str:
    if timeline.empty:
        return "N/A"
    trading_days = pd.to_datetime(timeline["trading_day"])
    return f"{trading_days.min().date()} -> {trading_days.max().date()}"


def _build_csv_links(report_stem: str) -> str:
    filenames = (
        f"{report_stem}_daily_timeline.csv",
        f"{report_stem}_monthly_summary.csv",
        f"{report_stem}_yearly_summary.csv",
        f"{report_stem}_account_summaries.csv",
        f"{report_stem}_events.csv",
    )
    return "\n".join(
        f"<li><a href=\"{html.escape(filename)}\">{html.escape(filename)}</a></li>"
        for filename in filenames
    )


def _build_chart_cards(
    timeline: pd.DataFrame,
    monthly_summary: pd.DataFrame,
) -> str:
    chart_specs = (
        (
            "Cumulative Net Cashflow",
            _build_cumulative_cashflow_chart(timeline),
        ),
        (
            "Active Account Counts",
            _build_active_accounts_chart(timeline),
        ),
        (
            "Payouts vs Costs",
            _build_payout_cost_chart(monthly_summary),
        ),
    )
    return "".join(
        (
            "<div class=\"chart-card\">"
            f"<h3>{html.escape(title)}</h3>"
            f"<img class=\"chart\" src=\"data:image/png;base64,{chart_b64}\" alt=\"{html.escape(title)}\" />"
            "</div>"
        )
        for title, chart_b64 in chart_specs
        if chart_b64
    )


def _build_cumulative_cashflow_chart(timeline: pd.DataFrame) -> str:
    if timeline.empty:
        return ""
    chart_frame = timeline.copy()
    chart_frame["trading_day"] = pd.to_datetime(chart_frame["trading_day"])
    fig, ax = plt.subplots(figsize=(7.0, 3.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(
        chart_frame["trading_day"],
        chart_frame["cumulative_net_cashflow"],
        color="#60a5fa",
        linewidth=2.0,
    )
    ax.set_title("Cumulative Net Cashflow", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    ax.grid(alpha=0.18, linestyle="--", color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout()
    return _figure_to_base64(fig)


def _build_active_accounts_chart(timeline: pd.DataFrame) -> str:
    if timeline.empty:
        return ""
    chart_frame = timeline.copy()
    chart_frame["trading_day"] = pd.to_datetime(chart_frame["trading_day"])
    fig, ax = plt.subplots(figsize=(7.0, 3.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(
        chart_frame["trading_day"],
        chart_frame["active_funded"],
        color="#34d399",
        linewidth=2.0,
        label="Active funded",
    )
    ax.plot(
        chart_frame["trading_day"],
        chart_frame["active_challenges"],
        color="#f59e0b",
        linewidth=2.0,
        label="Active challenges",
    )
    ax.set_title("Active Accounts", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    ax.grid(alpha=0.18, linestyle="--", color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("#cbd5e1")
    fig.tight_layout()
    return _figure_to_base64(fig)


def _build_payout_cost_chart(monthly_summary: pd.DataFrame) -> str:
    if monthly_summary.empty:
        return ""
    chart_frame = monthly_summary.tail(12).copy()
    x = range(len(chart_frame))
    fig, ax = plt.subplots(figsize=(7.0, 3.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.bar(
        [index - 0.2 for index in x],
        chart_frame["trader_payouts"],
        width=0.4,
        color="#60a5fa",
        label="Trader payouts",
    )
    ax.bar(
        [index + 0.2 for index in x],
        chart_frame["challenge_costs"],
        width=0.4,
        color="#f87171",
        label="Challenge costs",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(chart_frame["month"], rotation=45, ha="right", color="#cbd5e1")
    ax.set_title("Monthly Payouts vs Costs", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", colors="#cbd5e1", labelsize=9)
    ax.grid(axis="y", alpha=0.18, linestyle="--", color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("#cbd5e1")
    fig.tight_layout()
    return _figure_to_base64(fig)


def _figure_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _html_styles() -> str:
    return """
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0f172a;
  color: #e2e8f0;
}
.page {
  max-width: 1400px;
  margin: 0 auto;
  padding: 24px;
}
.hero {
  background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
  border: 1px solid #334155;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 24px;
}
.hero h1 {
  margin: 0 0 12px 0;
  font-size: 32px;
}
.hero-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  color: #94a3b8;
  font-size: 14px;
}
section {
  background: #111827;
  border: 1px solid #334155;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 24px;
}
section h2 {
  margin-top: 0;
  margin-bottom: 16px;
  font-size: 22px;
}
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
}
.card-grid.compact {
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.metric-card {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 14px 16px;
}
.metric-title {
  color: #94a3b8;
  font-size: 13px;
  margin-bottom: 8px;
}
.metric-value {
  color: #f8fafc;
  font-size: 22px;
  font-weight: 700;
}
.two-col {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 24px;
}
.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 18px;
}
.chart-card {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 14px;
}
.chart-card h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
  color: #f8fafc;
}
.chart {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 8px;
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid #334155;
  border-radius: 12px;
}
table.report-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  background: #0f172a;
}
table.report-table thead th {
  position: sticky;
  top: 0;
  background: #1e293b;
  color: #f8fafc;
  text-align: left;
  padding: 10px 12px;
  border-bottom: 1px solid #334155;
}
table.report-table tbody td {
  padding: 8px 12px;
  border-bottom: 1px solid #1f2937;
  color: #cbd5e1;
  white-space: nowrap;
}
table.report-table tbody tr:nth-child(even) {
  background: #111827;
}
table.report-table tbody tr:nth-child(odd) {
  background: #0b1220;
}
.empty-state {
  color: #94a3b8;
  margin: 0;
}
.file-list {
  margin: 0;
  padding-left: 18px;
}
.file-list a {
  color: #60a5fa;
  text-decoration: none;
}
.file-list a:hover {
  text-decoration: underline;
}
"""
