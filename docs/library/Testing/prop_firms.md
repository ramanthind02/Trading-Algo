# Prop Firm Simulation

> **Scope:** Research-side prop-firm portfolio simulation integrated into the portfolio test pipeline.

---

## Purpose

Prop-firm simulation runs automatically inside the portfolio test pipeline and produces
per-phase HTML/Markdown reports alongside the standard tearsheets.

The simulation layer lives entirely on the **research side** — there is no top-level
`prop_firms/` package. The simulation engine is provided by
`quantfoundry_core.prop_firm` (the `quantfoundry_core` dependency); the glue that ties
it into the research pipeline is:

- `research/portfolio/prop_firm_bridge.py` — converts `PhaseResult` daily returns and
  builds `PortfolioSimulationConfig` from `PropFirmReportConfig`.
- `research/portfolio/prop_firm_report_builder.py` — writes Markdown, HTML, and optional
  CSV artifacts from a `PortfolioSimulationResult`.
- `research/portfolio/prop_firm_reports.py` — top-level entry: `run_prop_firm_reports_for_phases`.
- `research/portfolio/portfolio_prop_firm_reports.py` — deprecated shim re-exporting the above.
- `research/portfolio/run_portfolio_prop_firm.py` — deprecated entrypoint stub (prints a
  migration message and exits 2; do not use).

---

## Architecture

```
PhaseResult  ──┐
               ├──► build_prop_firm_returns (prop_firm_bridge)
               │         │
               │         ▼
               │    pd.Series (daily simple returns)
               │         │
               │         ▼
               │    align_portfolio_returns_with_report_engine
               │    portfolio_simulation_config
               │         │
               ▼         ▼
CfdPortfolioSimulator.simulate()   ← create_prop_firm_portfolio_simulator
               │
               ├──► PortfolioSimulationResult
               │         │
               │    generate_portfolio_report (prop_firm_report_builder)
               │         │
               ▼         ▼
        PropFirmReportArtifacts  (Markdown, HTML, CSVs)
```

The simulator is loaded by `create_prop_firm_portfolio_simulator(firm_id)`, which calls
`quantfoundry_core.prop_firm.create_simulator_for_firm`. The default `firm_id` is
`"fundednext"` (set in `PropFirmReportConfig.firm_id`).

---

## Integration with the Portfolio Test Pipeline

After train / validation / test phases complete, `run_portfolio_test_pipeline` (in
`research/portfolio/pipelines/portfolio_test.py`) calls
`run_prop_firm_reports_for_phases` when
`PortfolioResearchConfig.prop_firm_report.enabled` is `True` (default in `load_config()`).

**Run the pipeline:**

```powershell
# Full portfolio test (includes prop-firm reports automatically)
.\.venv\Scripts\python.exe -m research.portfolio.run_portfolio_test
```

Or trigger via the frontend app (portfolio test stage automatically includes prop-firm reports).

**Disable prop-firm reports for a single run:**

```python
from dataclasses import replace
config = replace(config, prop_firm_report=replace(config.prop_firm_report, enabled=False))
```

---

## Configuration — `PropFirmReportConfig`

Defined in `research/portfolio/config.py`. Set via `PortfolioResearchConfig.prop_firm_report`
(a field on the portfolio config returned by `load_config()`).

| Field | Default | Notes |
|---|---|---|
| `enabled` | `True` | Set `False` to skip all reports |
| `firm_id` | `"fundednext"` | QF Core preset name |
| `phases` | `("train", "validation", "test")` | Which phases to report on |
| `output_subdir` | `"prop_firm"` | Under `{output_root}/{phase}/` |
| `account_code` | `"50000"` | Preset account size |
| `report_stem` | `"fundednext_portfolio_report"` | File basename |
| `save_csvs` | `True` | Write CSV artifacts |
| `funded_account_cap` | `10**9` (unlimited) | `None` defers to QF preset (typically 6) |
| `challenge_account_cap` | `6` | Max concurrent challenges |
| `challenges_per_purchase_window` | `1` | |
| `payout_buffer_amount` | `2500.0` | |
| `payout_withdrawal_fraction` | `1.0` | |
| `challenge_vol_multiplier` | `2.0` | Scales shared returns during challenge phase |
| `funded_vol_multiplier` | `0.5` | Scales shared returns during funded phase |
| `return_target_annual_volatility` | `0.10` | |
| `return_target_sharpe` | `2.0` | |
| `rolling_enabled` | `True` | Run `simulate_rolling` with rolling windows |
| `rolling_window_months` | `12` | Rolling window length in months |

---

## Report Artifacts

Reports are written under `{output_root}/{phase}/prop_firm/{firm_id}/`.

| File | Description |
|---|---|
| `{stem}_{phase}.md` | Markdown report |
| `{stem}_{phase}.html` | Self-contained HTML report with embedded charts |
| `{stem}_{phase}_daily_timeline.csv` | Daily simulation timeline |
| `{stem}_{phase}_monthly_summary.csv` | Monthly cashflow summary |
| `{stem}_{phase}_monthly_breakdown.csv` | QF Core monthly breakdown (when present) |
| `{stem}_{phase}_yearly_summary.csv` | Yearly cashflow summary |
| `{stem}_{phase}_account_summaries.csv` | Per-account outcome summary |
| `{stem}_{phase}_events.csv` | Event log |
| `{stem}_{phase}_rolling_pooled_monthly_stats.csv` | Pooled month-offset stats across windows |
| `{stem}_{phase}_rolling_batch_statistics.json` | Cross-window EV (`batch_statistics.to_record()`) |

Rolling artifacts are only written when `rolling_enabled=True` and the phase has
sufficient data (at least `rolling_window_months` calendar months).

**Rolling interpretation:** `month_offset` 1–2 are ramp-up (challenge costs dominate);
offsets 6–11 in the HTML/Markdown steady-state table approximate post-ramp monthly EV.

---

## `PropFirmReportArtifacts`

```python
@dataclass(frozen=True)
class PropFirmReportArtifacts:
    report_markdown_path: Path
    report_html_path: Path
    daily_timeline_csv_path: Path | None = None
    monthly_summary_csv_path: Path | None = None
    monthly_breakdown_csv_path: Path | None = None
    yearly_summary_csv_path: Path | None = None
    account_summaries_csv_path: Path | None = None
    events_csv_path: Path | None = None
    rolling_pooled_monthly_stats_csv_path: Path | None = None
    rolling_batch_statistics_json_path: Path | None = None
```

Returned by `generate_portfolio_report` in
`research/portfolio/prop_firm_report_builder.py`. CSV paths are `None` when `save_csvs=False`
or the data frame is empty.

---

## Calling Directly

`run_prop_firm_reports_for_phases` can be called outside the pipeline:

```python
from research.portfolio.prop_firm_reports import run_prop_firm_reports_for_phases

# phase_results: dict[str, PhaseResult] from run_portfolio_test_pipeline
artifacts = run_prop_firm_reports_for_phases(portfolio_config, phase_results)
# artifacts["test"].report_html_path, etc.
```

The standalone runner `research/portfolio/run_portfolio_prop_firm.py` is **deprecated**
and will exit with an error message directing you to `run_portfolio_test` instead.

---

## Tests

Unit tests for the report builder live in
`tests/unit-tests/portfolio_research/test_prop_firm_report_builder.py`.

They use `quantfoundry_core.prop_firm.create_simulator_for_firm("fundednext")` directly
with a synthetic return series — no pipeline fixtures required.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit-tests\portfolio_research\test_prop_firm_report_builder.py -v
```

---

## Payout Policy

`portfolio_simulation_config` in `prop_firm_bridge.py` wires payout policy to
`PortfolioPayoutPolicyMode.AGGRESSIVE` (request the maximum eligible payout as soon as
the account qualifies). The `payout_buffer_amount` and `payout_withdrawal_fraction`
fields on `PropFirmReportConfig` are passed through to
`PortfolioPayoutPolicyConfig`.

---

## Removed Items

The following items from the old `prop_firms/` top-level package **no longer exist**:

- `prop_firms/{apex,lucid,mffu,topstep,tradeday,fundednext}/` — deleted
- `prop_firms/base/` — deleted
- `prop_firms/run_apex_portfolio_report.py`, `run_lucid_portfolio_report.py`, `run_fundednext_portfolio_report.py` — deleted
- `prop_firms/run_lucid_hyperopt.py` — deleted
- `prop_firms/report_config.py`, `prop_firms/reporting.py` — deleted
- `python prop_firms/run_apex_portfolio_report.py` — use `python -m research.portfolio.run_portfolio_test` instead

> _Verified against current code via CodeGraph on 2026-06-07._
