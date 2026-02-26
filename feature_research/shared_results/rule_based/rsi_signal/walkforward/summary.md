# Walkforward Summary: rule_based/rsi_signal

## Overview
- Folds: 8
- Scored rows: 336
- Selection rows: 8
- Objective metric: t_stat

## Aggregated OOS Metrics
| metric | value |
| --- | --- |
| objective_metric_name | t_stat |
| num_folds | 8 |
| num_selected_rows | 8 |
| mean_selected_raw_t_stat | 0.6001212502415252 |
| median_selected_raw_t_stat | 0.6628195729175504 |
| std_selected_raw_t_stat | 0.8390519168464076 |
| min_selected_raw_t_stat | -0.708358787168111 |
| max_selected_raw_t_stat | 2.0405250472708794 |
| positive_raw_fold_rate | 0.75 |
| mean_selected_smoothed_t_stat | 2.0013835560079043 |
| unique_selected_features | 2 |
| most_selected_feature | exit_bars=5|exit_policy=threshold_or_bars|overbought=95|oversold=30|rsi_period=2|strategy_mode=long |
| most_selected_feature_count | 6 |
| mean_oos_portfolio_sharpe | 0.8761716293655059 |

## Aggregate Walkforward Test Performance
| metric | value |
| --- | --- |
| n_periods | 2014.0 |
| total_return | 1.789718790463016 |
| mean_return | 0.0006074498864013706 |
| std_return | 0.013939715753605378 |
| sharpe_annualized | 0.6917621686525935 |
| sortino_annualized | 0.532125077401422 |
| max_drawdown | 0.3715107981991993 |
| calmar_ratio | 4.817407190149536 |
| win_rate | 0.19910625620655412 |

## Model vs Oracle (Efficiency Ratio)
| metric | value |
| --- | --- |
| mean_efficiency_ratio | 1.0020288749179338 |
| median_efficiency_ratio | 1.0 |
| min_efficiency_ratio | 0.9988318338319924 |
| max_efficiency_ratio | 1.0129091998542923 |
| std_efficiency_ratio | 0.00440512737068783 |
| mean_wf_sharpe | 0.8761716293655059 |
| mean_oracle_sharpe | 0.8748783030168322 |

## Selected Parameters By Fold
| fold_id | param_label | raw_objective | oos_objective | smoothed_objective | rank | trade_frequency | selected_in_top_k | selected_long_bin | train_start | train_end | test_start | test_end | objective_metric_name | param_exit_bars | param_exit_policy | param_overbought | param_oversold | param_rsi_period | param_strategy_mode | context_aggregate_returns_last_date | context_last_fold_test_end | context_period_end | context_period_start | context_strategy | context_target_col | context_tickers | context_walkforward_num_steps | context_walkforward_selection_method | context_walkforward_test_step | context_walkforward_top_k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | exit_bars=5|exit_policy=threshold_or_bars|overbought=90|oversold=25|rsi_period=2|strategy_mode=long | 2.149740211053674 | -0.708358787168111 | 2.069573132118653 | 1 | 0.37177461822011587 | True | None | 2001-01-04 00:00:00 | 2015-12-31 00:00:00 | 2016-01-04 00:00:00 | 2016-12-30 00:00:00 | t_stat | 5 | threshold_or_bars | 90 | 25 | 2 | long | 2023-12-27 | 2023-12-27 | 2023-12-30 | 2000-01-01 | long | log_return_atr | ["ES","NQ","RTY"] | 8 | top_k | 365 | 1 |
| 1 | exit_bars=5|exit_policy=threshold_or_bars|overbought=90|oversold=25|rsi_period=2|strategy_mode=long | 2.3287854653814315 | 2.0405250472708794 | 2.2405099508951105 | 1 | 0.3652173913043478 | True | None | 2002-01-04 00:00:00 | 2016-12-30 00:00:00 | 2017-01-03 00:00:00 | 2017-12-29 00:00:00 | t_stat | 5 | threshold_or_bars | 90 | 25 | 2 | long | 2023-12-27 | 2023-12-27 | 2023-12-30 | 2000-01-01 | long | log_return_atr | ["ES","NQ","RTY"] | 8 | top_k | 365 | 1 |
| 2 | exit_bars=5|exit_policy=threshold_or_bars|overbought=95|oversold=30|rsi_period=2|strategy_mode=long | 2.3604271023474728 | 0.29117775555893827 | 2.349286590869565 | 1 | 0.38522427440633245 | True | None | 2003-01-06 00:00:00 | 2017-12-29 00:00:00 | 2018-01-02 00:00:00 | 2018-12-28 00:00:00 | t_stat | 5 | threshold_or_bars | 95 | 30 | 2 | long | 2023-12-27 | 2023-12-27 | 2023-12-30 | 2000-01-01 | long | log_return_atr | ["ES","NQ","RTY"] | 8 | top_k | 365 | 1 |
| 3 | exit_bars=5|exit_policy=threshold_or_bars|overbought=95|oversold=30|rsi_period=2|strategy_mode=long | 2.2602488668285114 | 0.16657277493225406 | 2.1588307958522 | 1 | 0.38469656992084433 | True | None | 2004-01-05 00:00:00 | 2018-12-28 00:00:00 | 2018-12-31 00:00:00 | 2019-12-30 00:00:00 | t_stat | 5 | threshold_or_bars | 95 | 30 | 2 | long | 2023-12-27 | 2023-12-27 | 2023-12-30 | 2000-01-01 | long | log_return_atr | ["ES","NQ","RTY"] | 8 | top_k | 365 | 1 |
| 4 | exit_bars=5|exit_policy=threshold_or_bars|overbought=95|oversold=30|rsi_period=2|strategy_mode=long | 2.176568838801844 | -0.29273768688622975 | 2.0199206338956186 | 1 | 0.3836810139952469 | True | None | 2005-01-03 00:00:00 | 2019-12-30 00:00:00 | 2019-12-31 00:00:00 | 2020-12-29 00:00:00 | t_stat | 5 | threshold_or_bars | 95 | 30 | 2 | long | 2023-12-27 | 2023-12-27 | 2023-12-30 | 2000-01-01 | long | log_return_atr | ["ES","NQ","RTY"] | 8 | top_k | 365 | 1 |
| 5 | exit_bars=5|exit_policy=threshold_or_bars|overbought=95|oversold=30|rsi_period=2|strategy_mode=long | 1.981884062887636 | 1.0344613902761626 | 1.8002575430653212 | 1 | 0.384310618066561 | True | None | 2006-01-03 00:00:00 | 2020-12-29 00:00:00 | 2020-12-30 00:00:00 | 2021-12-29 00:00:00 | t_stat | 5 | threshold_or_bars | 95 | 30 | 2 | long | 2023-12-27 | 2023-12-27 | 2023-12-30 | 2000-01-01 | long | log_return_atr | ["ES","NQ","RTY"] | 8 | top_k | 365 | 1 |
| 6 | exit_bars=5|exit_policy=threshold_or_bars|overbought=95|oversold=30|rsi_period=2|strategy_mode=long | 1.8624062242667785 | 1.0874973417940428 | 1.5905334956755455 | 1 | 0.38104040137311856 | True | None | 2007-01-03 00:00:00 | 2021-12-29 00:00:00 | 2021-12-30 00:00:00 | 2022-12-29 00:00:00 | t_stat | 5 | threshold_or_bars | 95 | 30 | 2 | long | 2023-12-27 | 2023-12-27 | 2023-12-30 | 2000-01-01 | long | log_return_atr | ["ES","NQ","RTY"] | 8 | top_k | 365 | 1 |
| 7 | exit_bars=5|exit_policy=threshold_or_bars|overbought=95|oversold=30|rsi_period=2|strategy_mode=long | 2.007448749948902 | 1.181832166154265 | 1.7821563056912175 | 1 | 0.38589540412044376 | True | None | 2008-01-03 00:00:00 | 2022-12-29 00:00:00 | 2022-12-30 00:00:00 | 2023-12-27 00:00:00 | t_stat | 5 | threshold_or_bars | 95 | 30 | 2 | long | 2023-12-27 | 2023-12-27 | 2023-12-30 | 2000-01-01 | long | log_return_atr | ["ES","NQ","RTY"] | 8 | top_k | 365 | 1 |

## Fold Timeline (Tabular)
| fold_id | train_start | train_end | test_start | test_end | train_samples | test_samples |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2001-01-04 00:00:00 | 2015-12-31 00:00:00 | 2016-01-04 00:00:00 | 2016-12-30 00:00:00 | 3798 | 252 |
| 1 | 2002-01-04 00:00:00 | 2016-12-30 00:00:00 | 2017-01-03 00:00:00 | 2017-12-29 00:00:00 | 3795 | 252 |
| 2 | 2003-01-06 00:00:00 | 2017-12-29 00:00:00 | 2018-01-02 00:00:00 | 2018-12-28 00:00:00 | 3790 | 251 |
| 3 | 2004-01-05 00:00:00 | 2018-12-28 00:00:00 | 2018-12-31 00:00:00 | 2019-12-30 00:00:00 | 3790 | 252 |
| 4 | 2005-01-03 00:00:00 | 2019-12-30 00:00:00 | 2019-12-31 00:00:00 | 2020-12-29 00:00:00 | 3787 | 252 |
| 5 | 2006-01-03 00:00:00 | 2020-12-29 00:00:00 | 2020-12-30 00:00:00 | 2021-12-29 00:00:00 | 3786 | 253 |
| 6 | 2007-01-03 00:00:00 | 2021-12-29 00:00:00 | 2021-12-30 00:00:00 | 2022-12-29 00:00:00 | 3787 | 252 |
| 7 | 2008-01-03 00:00:00 | 2022-12-29 00:00:00 | 2022-12-30 00:00:00 | 2023-12-27 00:00:00 | 3786 | 250 |

## Portfolio Simulation (Stage 2)
- Mean OOS Portfolio Sharpe: 0.876172
| fold_id | oos_portfolio_sharpe | n_params_selected | error |
| --- | --- | --- | --- |
| 0 | -0.5613555010116112 | 1 |  |
| 1 | 2.266191487013032 | 1 |  |
| 2 | -0.03448913463962509 | 1 |  |
| 3 | 1.5394591402029696 | 1 |  |
| 4 | -0.07998574947152874 | 1 |  |
| 5 | 1.0612529665078658 | 1 |  |
| 6 | 0.6078060915063452 | 1 |  |
| 7 | 2.2104937348165987 | 1 |  |

## Notes
- **Readable tables**: open **`tables_report.html`** for all tabular data in one page (nav by section, scrollable tables).
- **Tabular data** (CSVs) are under `tables/`: folds, fold_scores, selection_summary, selected_params_detailed, selection_params_and_regions, fold_signal_metrics, aggregate_ensemble_metrics, oos_metrics.
- **Tearsheets** (QuantStats HTML reports) are under `tearsheets/`: aggregate walkforward ensemble plus per-fold ensemble and per-param-combo reports when portfolio simulation runs. Strategy returns cover only the union of fold test periods; **last fold test end** (see `context_last_fold_test_end` in selected_params_detailed or report.json) is the actual coverage end. Years with no test data (e.g. after the last fold) show as NaN in the tearsheet.
- Use `tables/oos_metrics.csv` for aggregate metrics, `tables/selected_params_detailed.csv` for parameter-level analysis, `tables/fold_scores.csv` for full ranking data per fold.
- **Actual test coverage**: through 2023-12-27 (last fold test_end). Config period_end may be later; re-run with data through that date to extend coverage.
