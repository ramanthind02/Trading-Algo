# Walkforward Summary: rule_based/seasonal_indices_eof

## Overview
- Folds: 1
- Scored rows: 1
- Selection rows: 1
- Objective metric: t_stat

## Aggregated OOS Metrics
| metric | value |
| --- | --- |
| objective_metric_name | t_stat |
| num_folds | 1 |
| num_selected_rows | 1 |
| mean_selected_raw_t_stat | 0.13985420553879274 |
| median_selected_raw_t_stat | 0.13985420553879274 |
| std_selected_raw_t_stat | 0.0 |
| min_selected_raw_t_stat | 0.13985420553879274 |
| max_selected_raw_t_stat | 0.13985420553879274 |
| positive_raw_fold_rate | 1.0 |
| mean_selected_smoothed_t_stat | -0.8867140962278429 |
| unique_selected_features | 1 |
| most_selected_feature |  |
| most_selected_feature_count | 1 |
| mean_oos_portfolio_sharpe | 1.2637383968945735 |

## Aggregate Walkforward Test Performance
| metric | value |
| --- | --- |
| n_periods | 1261.0 |
| total_return | 0.5345810396964996 |
| mean_return | 0.0004051674066848667 |
| std_return | 0.011398972175530079 |
| sharpe_annualized | 0.5642467658996337 |
| sortino_annualized | 0.4211786424916147 |
| max_drawdown | 0.2722377054486907 |
| calmar_ratio | 1.963655397460193 |
| win_rate | 0.19349722442505948 |

## Selected Parameters By Fold
| fold_id | param_label | raw_objective | oos_objective | smoothed_objective | rank | trade_frequency | selected_in_top_k | selected_long_bin | train_start | train_end | test_start | test_end | objective_metric_name | context_aggregate_returns_last_date | context_last_fold_test_end | context_period_end | context_period_start | context_strategy | context_target_col | context_tickers | context_validation_selection_method | context_validation_test_end | context_validation_test_start | context_validation_top_k | context_validation_train_end | context_validation_train_start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 |  | -0.8867140962278429 | 0.13985420553879274 | -0.8867140962278429 | 1 | 0.29482683033757123 | True | None | 2000-01-03 00:00:00 | 2017-12-29 00:00:00 | 2018-01-02 00:00:00 | 2022-12-30 00:00:00 | t_stat | 2022-12-30 | 2022-12-30 | 2022-12-31 | 2000-01-01 | long | log_return_atr | ["NQ","YM","RTY","ES"] | top_k | 2022-12-31 | 2018-01-01 | 1 | 2017-12-31 | 2000-01-01 |

## Fold Timeline (Tabular)
| fold_id | train_start | train_end | test_start | test_end | train_samples | test_samples |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2000-01-03 00:00:00 | 2017-12-29 00:00:00 | 2018-01-02 00:00:00 | 2022-12-30 00:00:00 | 4562 | 1261 |

## Portfolio Simulation (Stage 2)
- Mean OOS Portfolio Sharpe: 1.263738
| fold_id | oos_portfolio_sharpe | n_params_selected | error |
| --- | --- | --- | --- |
| 0 | 1.2637383968945735 | 1 |  |

## Notes
- **Readable tables**: open **`tables_report.html`** for all tabular data in one page (nav by section, scrollable tables).
- **Tabular data** (CSVs) are under `tables/`: folds, fold_scores, selection_summary, selected_params_detailed, selection_params_and_regions, fold_signal_metrics, aggregate_ensemble_metrics, oos_metrics.
- **Tearsheets** (QuantStats HTML reports) are under `tearsheets/`: aggregate walkforward ensemble plus per-fold ensemble and per-param-combo reports when portfolio simulation runs. Strategy returns cover only the union of fold test periods; **last fold test end** (see `context_last_fold_test_end` in selected_params_detailed or report.json) is the actual coverage end. Years with no test data (e.g. after the last fold) show as NaN in the tearsheet.
- Use `tables/oos_metrics.csv` for aggregate metrics, `tables/selected_params_detailed.csv` for parameter-level analysis, `tables/fold_scores.csv` for full ranking data per fold.
- **Actual test coverage**: through 2022-12-30 (last fold test_end). Config period_end may be later; re-run with data through that date to extend coverage.
