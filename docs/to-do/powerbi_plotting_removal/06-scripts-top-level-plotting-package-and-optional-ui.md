# Ticket 06: Scripts, Top-Level `plotting/`, And Optional Chart UI

## Why

Demos and the top-level `plotting/` package exist mainly to produce PNG/HTML for humans. Most of this is outside the **keep** list (tearsheets, prop-firm HTML, **Norgate migration Plotly HTML**).

## Goal

Remove or quarantine plotting-only scripts and the non–prop-firm `plotting/` helpers; decide explicitly what to do about the **Flask chart viewer**.

## Scope

- **Scripts (likely remove or no-op):**
  - `scripts/demo_robustness_test.py`
  - `scripts/demo_prop_firm_simulator.py` (uses `plotting/prop_firm.py` for PNG “dashboards”—**not** the same as `prop_firms/*` HTML reports; safe to remove with demos unless product wants to keep simulator **without** plots)
- **Keep (do not strip plotting):**
  - `scripts/validate_norgate_migration.py` — retain Plotly `write_html` (and any related chart outputs) for Norgate data migration QA; still emit JSON/CSV summaries alongside if the script already does.
- **Top-level package `plotting/`:** remove `plotting/prop_firm.py`, `plotting/robustness.py`, and any `__init__.py` exports that only served demos—**after** confirming no production import paths (grep `from plotting.` / `import plotting`).
- **Optional product decision — `frontend/`:** `frontend/app.py` + `frontend/index.html` serve a candle **Chart Viewer** (TradingView-style). This is not matplotlib but is an in-repo chart UI. Record decision: **keep** for operational debugging, **remove** if all charting moves to Power BI + external tools, or **move** to a separate repo.

## Out Of Scope

- `prop_firms/reporting.py` and related HTML report runners (keep-list).
- `scripts/validate_norgate_migration.py` plotting (keep-list).

## Acceptance Criteria

- No demo script in `scripts/` fails CI because of missing plot backends after removal.
- Norgate migration docs continue to describe Plotly HTML report outputs (unchanged contract).
- Explicit one-line decision recorded in ticket 07 or `README` for `frontend/` chart viewer.

## Risks

- If `plotly` is dropped elsewhere, ticket **07** must still retain it as long as Norgate migration QA depends on it.

## Dependencies

- **05** if robustness plotting shared code paths.
