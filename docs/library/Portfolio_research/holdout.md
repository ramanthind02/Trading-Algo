# Portfolio Holdout Research

Local command surface for project test-zone evaluation and monitoring.

## Command

```bash
python -m research.portfolio holdout
```

Runs, in order:

1. Two-fold rolling (default) or single-fit portfolio evaluation
2. Per-strategy holdout monitoring (validation-parity charts)
3. Portfolio holdout analytics per `docs/SaaS/robustness_tests/portfolio_holdout.md`

## Configuration

Edit `research/portfolio/config.py`:

- `portfolio_fit_mode`: `ROLLING_HOLDOUT` (default) or `SINGLE_FIT`
- `ROLLING_HOLDOUT` runs **two** folds only:
  - **validation**: fit on `train_window`, score `validation_window`
  - **holdout_test**: fit on a train-length window rolled to end at `validation_window.end`, score `test_window`
- Holdout monitoring CSVs use the **test** fold only (not stitched validation segments).
- `num_steps` / `test_window_years` still align walkforward window math with feature research; they do not tile the test window into many folds.
- `holdout_robustness`: monitoring thresholds (CUSUM, rolling Sharpe, bootstrap settings)

## Artifact layout

```text
<output_root>/holdout/
  fold_manifest.json
  returns/
    strategy_returns_research.csv
    strategy_returns_holdout.csv
    portfolio_returns_holdout.csv
  validation/          # fold 0: train fit → validation score
  test/                # fold 1: rolled train fit → test score
    portfolio/
    weight_layer_weights_holdout.csv
  strategies/<ensemble>/
    holdout_robustness_report.json
    matplotlib/*.png
  visualization/portfolio/
    portfolio_holdout_report.json
    correlation_matrix_*.csv
    matplotlib/*.png
```

## UI

The primary UI is the **React/Vite frontend** (`frontend/web/`) served by the FastAPI backend
(`frontend/api/server.py`). Navigate to the **Portfolio research** section (`/portfolio` route)
in the running dev server (port 5173) or the built app. Phase tabs (Research / Validation /
Holdout) mirror the docs workflow: Portfolio Test → Strategy Holdout → Portfolio Holdout.

```bash
# backend
.\.venv\Scripts\python.exe -m uvicorn frontend.api.server:app --reload --port 5057
# frontend dev server
cd frontend\web && npm run dev   # proxies /api → 5057, served at localhost:5173
```

Artifact panels are organized by the React frontend under the Portfolio research page; typical
sections include tearsheets, robustness summaries, weight-layer weights, and return matrices.

> **Note:** A legacy Flask workspace (`frontend/app.py` → `/portfolio-research`) also exists but
> is superseded by the React/Vite app above for new work.

> _Verified against current code via CodeGraph on 2026-06-07._
