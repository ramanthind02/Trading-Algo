from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import html
from pathlib import Path

import pandas as pd

from prop_firms.optimization import LucidHyperoptResult


@dataclass(frozen=True)
class PropFirmOptimizationArtifacts:
    """Paths written by the optimization report generator."""

    report_markdown_path: Path
    report_html_path: Path
    trials_csv_path: Path | None = None
    seed_runs_csv_path: Path | None = None


def generate_optimization_report(
    result: LucidHyperoptResult,
    output_dir: Path,
    report_stem: str,
    save_csvs: bool = True,
) -> PropFirmOptimizationArtifacts:
    """Write readable artifacts for the optimizer run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_markdown_path = output_dir / f"{report_stem}.md"
    report_html_path = output_dir / f"{report_stem}.html"
    report_markdown_path.write_text(
        _build_markdown_report(result=result),
        encoding="utf-8",
    )
    report_html_path.write_text(
        _build_html_report(result=result, report_stem=report_stem),
        encoding="utf-8",
    )

    if not save_csvs:
        return PropFirmOptimizationArtifacts(
            report_markdown_path=report_markdown_path,
            report_html_path=report_html_path,
        )

    trials_csv_path = output_dir / f"{report_stem}_trials.csv"
    seed_runs_csv_path = output_dir / f"{report_stem}_seed_runs.csv"
    result.trials_frame.to_csv(trials_csv_path, index=False)
    result.seed_runs_frame.to_csv(seed_runs_csv_path, index=False)
    return PropFirmOptimizationArtifacts(
        report_markdown_path=report_markdown_path,
        report_html_path=report_html_path,
        trials_csv_path=trials_csv_path,
        seed_runs_csv_path=seed_runs_csv_path,
    )


def _build_markdown_report(result: LucidHyperoptResult) -> str:
    best_trial = result.best_trial
    stats = best_trial.batch_statistics
    top_trials = _format_markdown_table(result.trials_frame.head(10))
    top_seed_runs = _format_markdown_table(
        result.seed_runs_frame[
            result.seed_runs_frame["trial_number"] == best_trial.trial_number
        ].head(20)
    )
    return "\n".join(
        [
            f"# Lucid Hyperopt Report: {best_trial.simulation_config.account_code}",
            "",
            "## Best Trial",
            f"- Trial number: `{best_trial.trial_number}`",
            f"- Objective: `${best_trial.objective_value:,.2f}` average net cashflow",
            f"- Challenge vol multiplier: `{best_trial.simulation_config.challenge_vol_multiplier:.4f}`",
            f"- Funded vol multiplier: `{best_trial.simulation_config.funded_vol_multiplier:.4f}`",
            f"- Payout mode: `{best_trial.simulation_config.payout_policy.mode.value}`",
            f"- Buffer amount: `${best_trial.simulation_config.payout_policy.buffer_amount:,.2f}`",
            f"- Withdrawal fraction: `{best_trial.simulation_config.payout_policy.withdrawal_fraction:.4f}`",
            (
                "- Max payouts per funded account: "
                f"`{best_trial.simulation_config.max_payouts_per_funded_account}`"
            ),
            (
                "- Challenges per purchase window: "
                f"`{best_trial.simulation_config.purchase_policy.challenges_per_purchase_window}`"
            ),
            "",
            "## Monte Carlo Stats",
            f"- Runs per trial: `{result.config.monte_carlo_runs}`",
            f"- Trials searched: `{result.config.n_trials}`",
            f"- Expected trader payouts: `${stats.expected_total_trader_payouts:,.2f}`",
            (
                "- Probability of negative net cashflow: "
                f"`{stats.probability_negative_net_cashflow:.2%}`"
            ),
            (
                "- Expected funded accounts created: "
                f"`{stats.expected_funded_accounts_created:.2f}`"
            ),
            (
                "- Expected days to first payout: "
                f"`{stats.expected_days_to_first_payout}`"
            ),
            "",
            "## Top Trials",
            top_trials,
            "",
            "## Best Trial Seed Runs",
            top_seed_runs,
            "",
        ]
    )


def _build_html_report(result: LucidHyperoptResult, report_stem: str) -> str:
    best_trial = result.best_trial
    stats = best_trial.batch_statistics
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    top_trials = result.trials_frame.head(10).reset_index(drop=True)
    top_seed_runs = result.seed_runs_frame[
        result.seed_runs_frame["trial_number"] == best_trial.trial_number
    ].reset_index(drop=True)
    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\" />",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            (
                f"<title>{html.escape(f'Lucid Hyperopt Report: {best_trial.simulation_config.account_code}')}</title>"
            ),
            "<style>",
            _html_styles(),
            "</style>",
            "</head>",
            "<body>",
            "<div class=\"page\">",
            "<header class=\"hero\">",
            (
                f"<h1>Lucid Hyperopt Report: {html.escape(best_trial.simulation_config.account_code)}</h1>"
            ),
            "<div class=\"hero-meta\">",
            f"<span>Generated: {html.escape(generated_at)}</span>",
            f"<span>Report stem: {html.escape(report_stem)}</span>",
            f"<span>Trials: {result.config.n_trials}</span>",
            f"<span>Monte Carlo runs: {result.config.monte_carlo_runs}</span>",
            "</div>",
            "</header>",
            "<section>",
            "<h2>Best Trial</h2>",
            "<div class=\"card-grid\">",
            _metric_card("Trial number", str(best_trial.trial_number)),
            _metric_card(
                "Objective",
                f"${best_trial.objective_value:,.2f}",
            ),
            _metric_card(
                "Challenge vol",
                f"{best_trial.simulation_config.challenge_vol_multiplier:.4f}",
            ),
            _metric_card(
                "Funded vol",
                f"{best_trial.simulation_config.funded_vol_multiplier:.4f}",
            ),
            _metric_card(
                "Payout mode",
                best_trial.simulation_config.payout_policy.mode.value,
            ),
            _metric_card(
                "Purchase window",
                str(
                    best_trial.simulation_config.purchase_policy.challenges_per_purchase_window
                ),
            ),
            "</div>",
            "</section>",
            "<section>",
            "<h2>Best Trial Stats</h2>",
            "<div class=\"card-grid compact\">",
            _metric_card(
                "Expected trader payouts",
                f"${stats.expected_total_trader_payouts:,.2f}",
            ),
            _metric_card(
                "Negative cashflow probability",
                f"{stats.probability_negative_net_cashflow:.2%}",
            ),
            _metric_card(
                "Expected funded accounts",
                f"{stats.expected_funded_accounts_created:.2f}",
            ),
            _metric_card(
                "Expected first payout days",
                str(stats.expected_days_to_first_payout),
            ),
            "</div>",
            "</section>",
            "<section>",
            "<h2>Top Trials</h2>",
            _render_html_table(top_trials),
            "</section>",
            "<section>",
            "<h2>Best Trial Seed Runs</h2>",
            _render_html_table(top_seed_runs),
            "</section>",
            "</div>",
            "</body>",
            "</html>",
        ]
    )


def _format_markdown_table(frame: pd.DataFrame) -> str:
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
"""
