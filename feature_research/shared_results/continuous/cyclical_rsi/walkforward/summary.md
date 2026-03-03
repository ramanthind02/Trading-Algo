# Walkforward Summary: continuous/cyclical_rsi

## Overview
- Folds: 8
- Scored rows: 8
- Selection rows: 8
- Objective metric: t_stat

## Aggregated OOS Metrics
| metric | value |
| --- | --- |
| objective_metric_name | t_stat |
| num_folds | 8 |
| num_selected_rows | 8 |
| mean_selected_raw_t_stat | 0.8943095260406511 |
| median_selected_raw_t_stat | 0.8011641429571907 |
| std_selected_raw_t_stat | 1.0780088121097178 |
| min_selected_raw_t_stat | -0.3970880747745282 |
| max_selected_raw_t_stat | 3.3472567922322414 |
| positive_raw_fold_rate | 0.875 |
| mean_selected_smoothed_t_stat | 4.715763395120307 |
| unique_selected_features | 1 |
| most_selected_feature | bin_count=10|long_period=120|rsi_period=2|selected_bin=0|short_period=4 |
| most_selected_feature_count | 8 |
| mean_oos_portfolio_sharpe | 0.9381896655516578 |

## Aggregate Walkforward Test Performance
| metric | value |
| --- | --- |
| n_periods | 2013.0 |
| total_return | 1.4395434666027738 |
| mean_return | 0.0005328048597957811 |
| std_return | 0.013335121511002392 |
| sharpe_annualized | 0.6342660568258937 |
| sortino_annualized | 0.26040577837752094 |
| max_drawdown | 0.290063057502167 |
| calmar_ratio | 4.962863864840903 |
| win_rate | 0.06507699950322901 |

## Selected Parameters By Fold
| fold_id | param_label | raw_objective | oos_objective | smoothed_objective | rank | trade_frequency | selected_in_top_k | selected_long_bin | train_start | train_end | test_start | test_end | objective_metric_name | param_bin_count | param_long_period | param_rsi_period | param_selected_bin | param_short_period | context_aggregate_returns_last_date | context_last_fold_test_end | context_period_end | context_period_start | context_strategy | context_target_col | context_tickers | context_walkforward_num_steps | context_walkforward_selection_method | context_walkforward_test_step | context_walkforward_top_k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | bin_count=10|long_period=120|rsi_period=2|selected_bin=0|short_period=4 | 4.930298973855113 | 1.104488335921138 | 4.930298973855113 | 1 | 0.09848391470479477 | True | 0 | 2000-01-03 00:00:00 | 2015-12-31 00:00:00 | 2016-01-04 00:00:00 | 2016-12-30 00:00:00 | t_stat | 10 | 120 | 2 | 0 | 4 | 2023-12-27 | 2023-12-27 | 2023-12-31 | 2000-01-01 | long | log_return_atr | ["ES","NQ"] | 8 | marginal_peak | 365 | 1 |
| 1 | bin_count=10|long_period=120|rsi_period=2|selected_bin=0|short_period=4 | 4.8169266683503835 | 3.3472567922322414 | 4.8169266683503835 | 1 | 0.09848204368752314 | True | 0 | 2001-01-02 00:00:00 | 2016-12-30 00:00:00 | 2017-01-03 00:00:00 | 2017-12-29 00:00:00 | t_stat | 10 | 120 | 2 | 0 | 4 | 2023-12-27 | 2023-12-27 | 2023-12-31 | 2000-01-01 | long | log_return_atr | ["ES","NQ"] | 8 | marginal_peak | 365 | 1 |
| 2 | bin_count=10|long_period=120|rsi_period=2|selected_bin=0|short_period=4 | 4.97459161890731 | 0.05166470975705764 | 4.97459161890731 | 1 | 0.09843151784611584 | True | 0 | 2001-12-31 00:00:00 | 2017-12-29 00:00:00 | 2018-01-02 00:00:00 | 2018-12-31 00:00:00 | t_stat | 10 | 120 | 2 | 0 | 4 | 2023-12-27 | 2023-12-27 | 2023-12-31 | 2000-01-01 | long | log_return_atr | ["ES","NQ"] | 8 | marginal_peak | 365 | 1 |
| 3 | bin_count=10|long_period=120|rsi_period=2|selected_bin=0|short_period=4 | 4.972551919690044 | 1.3473223578436715 | 4.972551919690044 | 1 | 0.098676888833931 | True | 0 | 2002-12-31 00:00:00 | 2018-12-31 00:00:00 | 2019-01-02 00:00:00 | 2019-12-31 00:00:00 | t_stat | 10 | 120 | 2 | 0 | 4 | 2023-12-27 | 2023-12-27 | 2023-12-31 | 2000-01-01 | long | log_return_atr | ["ES","NQ"] | 8 | marginal_peak | 365 | 1 |
| 4 | bin_count=10|long_period=120|rsi_period=2|selected_bin=0|short_period=4 | 4.726916861940866 | 0.8332656800732768 | 4.726916861940866 | 1 | 0.09892419933226165 | True | 0 | 2003-12-31 00:00:00 | 2019-12-31 00:00:00 | 2020-01-02 00:00:00 | 2020-12-30 00:00:00 | t_stat | 10 | 120 | 2 | 0 | 4 | 2023-12-27 | 2023-12-27 | 2023-12-31 | 2000-01-01 | long | log_return_atr | ["ES","NQ"] | 8 | marginal_peak | 365 | 1 |
| 5 | bin_count=10|long_period=120|rsi_period=2|selected_bin=0|short_period=4 | 4.558561666031141 | 0.0985038014312467 | 4.558561666031141 | 1 | 0.09912139586684816 | True | 0 | 2004-12-30 00:00:00 | 2020-12-30 00:00:00 | 2020-12-31 00:00:00 | 2021-12-30 00:00:00 | t_stat | 10 | 120 | 2 | 0 | 4 | 2023-12-27 | 2023-12-27 | 2023-12-31 | 2000-01-01 | long | log_return_atr | ["ES","NQ"] | 8 | marginal_peak | 365 | 1 |
| 6 | bin_count=10|long_period=120|rsi_period=2|selected_bin=0|short_period=4 | 4.739596754151511 | 0.7690626058411048 | 4.739596754151511 | 1 | 0.09925742574257426 | True | 0 | 2005-12-30 00:00:00 | 2021-12-30 00:00:00 | 2021-12-31 00:00:00 | 2022-12-30 00:00:00 | t_stat | 10 | 120 | 2 | 0 | 4 | 2023-12-27 | 2023-12-27 | 2023-12-31 | 2000-01-01 | long | log_return_atr | ["ES","NQ"] | 8 | marginal_peak | 365 | 1 |
| 7 | bin_count=10|long_period=120|rsi_period=2|selected_bin=0|short_period=4 | 4.006662698036089 | -0.3970880747745282 | 4.006662698036089 | 1 | 0.09925742574257426 | True | 0 | 2007-01-02 00:00:00 | 2022-12-30 00:00:00 | 2023-01-03 00:00:00 | 2023-12-27 00:00:00 | t_stat | 10 | 120 | 2 | 0 | 4 | 2023-12-27 | 2023-12-27 | 2023-12-31 | 2000-01-01 | long | log_return_atr | ["ES","NQ"] | 8 | marginal_peak | 365 | 1 |

## Fold Timeline (Tabular)
| fold_id | train_start | train_end | test_start | test_end | train_samples | test_samples |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2000-01-03 00:00:00 | 2015-12-31 00:00:00 | 2016-01-04 00:00:00 | 2016-12-30 00:00:00 | 8113 | 504 |
| 1 | 2001-01-02 00:00:00 | 2016-12-30 00:00:00 | 2017-01-03 00:00:00 | 2017-12-29 00:00:00 | 8103 | 502 |
| 2 | 2001-12-31 00:00:00 | 2017-12-29 00:00:00 | 2018-01-02 00:00:00 | 2018-12-31 00:00:00 | 8097 | 504 |
| 3 | 2002-12-31 00:00:00 | 2018-12-31 00:00:00 | 2019-01-02 00:00:00 | 2019-12-31 00:00:00 | 8087 | 504 |
| 4 | 2003-12-31 00:00:00 | 2019-12-31 00:00:00 | 2020-01-02 00:00:00 | 2020-12-30 00:00:00 | 8087 | 504 |
| 5 | 2004-12-30 00:00:00 | 2020-12-30 00:00:00 | 2020-12-31 00:00:00 | 2021-12-30 00:00:00 | 8081 | 506 |
| 6 | 2005-12-30 00:00:00 | 2021-12-30 00:00:00 | 2021-12-31 00:00:00 | 2022-12-30 00:00:00 | 8080 | 504 |
| 7 | 2007-01-02 00:00:00 | 2022-12-30 00:00:00 | 2023-01-03 00:00:00 | 2023-12-27 00:00:00 | 8080 | 498 |

## Portfolio Simulation (Stage 2)
- Mean OOS Portfolio Sharpe: 0.938190
| fold_id | oos_portfolio_sharpe | n_params_selected | error |
| --- | --- | --- | --- |
| 0 | 1.4134049327719147 | 1 |  |
| 1 | 2.0539195265067356 | 1 |  |
| 2 | 0.34600619198532157 | 1 |  |
| 3 | 0.4825213540319828 | 1 |  |
| 4 | 0.464876756600926 | 1 |  |
| 5 | 1.5511779408902437 | 1 |  |
| 6 | 0.8431881018191767 | 1 |  |
| 7 | 0.35042251980696176 | 1 |  |

## Notes
- **Readable tables**: open **`tables_report.html`** for all tabular data in one page (nav by section, scrollable tables).
- **Tabular data** (CSVs) are under `tables/`: folds, fold_scores, selection_summary, selected_params_detailed, selection_params_and_regions, fold_signal_metrics, aggregate_ensemble_metrics, oos_metrics.
- **Tearsheets** (QuantStats HTML reports) are under `tearsheets/`: aggregate walkforward ensemble plus per-fold ensemble and per-param-combo reports when portfolio simulation runs. Strategy returns cover only the union of fold test periods; **last fold test end** (see `context_last_fold_test_end` in selected_params_detailed or report.json) is the actual coverage end. Years with no test data (e.g. after the last fold) show as NaN in the tearsheet.
- Use `tables/oos_metrics.csv` for aggregate metrics, `tables/selected_params_detailed.csv` for parameter-level analysis, `tables/fold_scores.csv` for full ranking data per fold.
- **Actual test coverage**: through 2023-12-27 (last fold test_end). Config period_end may be later; re-run with data through that date to extend coverage.
