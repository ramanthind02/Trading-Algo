# Portfolio Holdout Research

Local command surface for project test-zone evaluation and monitoring.

## Command

```bash
python -m portfolio_research holdout
```

Runs, in order:

1. Two-fold rolling (default) or single-fit portfolio evaluation
2. Per-strategy holdout monitoring (validation-parity charts)
3. Portfolio holdout analytics per `docs/SaaS/robustness_tests/portfolio_holdout.md`

## Configuration

Edit `portfolio_research/config.py`:

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

```text
frontend/app.py  →  /portfolio-research
```

Phases mirror the docs workflow: Portfolio Test → Strategy Holdout → Portfolio Holdout.

The workspace dashboard (shared with feature research via `frontend/static/research_workspace.js`) organizes artifacts into sections:

| Section | Typical artifacts |
|---------|-------------------|
| QuantStats Tearsheets | `*tearsheet*.html` under `holdout/validation` and `holdout/test` |
| Matplotlib Reports | `**/matplotlib/*.png` |
| Robustness Summaries | `holdout_robustness_report.json`, related CSV/MD per strategy |
| Portfolio Analytics | `portfolio_holdout_report.json`, correlation/contribution CSVs |
| Weight Layer | `weight_layer_weights*.csv` |
| Returns | `holdout/returns/*.csv` |
| Fold Manifest | `fold_manifest.json` |
| Raw data | CSV/JSON exports not shown as primary panels |

Use phase tabs and section navigation to preview tearsheets (embed), summary JSON panels, and tables without hunting filenames in a flat list.

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
