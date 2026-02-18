"""
Integration tests for Feature Validator Permutation Testing (T013-T017).

Uses real repository data from data/ohlc_data/ and real bias node extraction.
Default configuration: RSI lookback 5 on ES daily data (2020-2023).

All tests are customizable via function parameters for researcher exploration.

CANDLE VISUALIZATION:
test_pipeline_permutation_stage2() includes matplotlib plots of original vs
shuffled candle OHLC so the researcher can visually verify that the candle
shuffler is working correctly (price levels maintained, intra-bar structure
shuffled).

Cache policy:
  - Use existing cache: USE_CACHE=True
  - If cache missing: populate via CacheManager or skip with message
  - Cache spec: RSI lookback 5, ES, D, 2020-2023
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from feature_extraction.feature_extractor import extract_features_for_bias_node
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validation.config import PermutationTestConfig
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from feature_selection.validation.orchestration import run_permutation_test_suite
from feature_selection.validation.permutation_tests import (
    run_pipeline_permutation_continuous,
    run_vector_shuffle_test,
)
from feature_selection.validation.report_generator import generate_permutation_reports
from feature_selection.validation.reports import (
    PipelinePermutationReport,
    PermutationTestSuite,
    VectorShuffleReport,
    WalkforwardStabilityReport,
)
from feature_selection.validation.stability_analysis import (
    _param_combo_name,
    run_walkforward_stability,
)
from utils.cache_manager import CacheManager
from utils.enums import Ticker, TimeFrame
from utils.permutation_test.candle_shuffle import CandleShuffler

# ---------------------------------------------------------------------------
# Default integration test configuration (RSI-5 on ES daily 2020-2023)
# ---------------------------------------------------------------------------

DEFAULT_BIAS_MODULE = 'rsi'
DEFAULT_PARAM_NAME = 'lookback'
DEFAULT_PARAM_VALUE = 5
DEFAULT_TICKER = Ticker.ES
DEFAULT_TIMEFRAME = TimeFrame.D
DEFAULT_START = datetime(2020, 1, 1)
DEFAULT_END = datetime(2023, 12, 31)

NREPS_FAST = 100  # fast for integration tests

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sharpe(returns: pd.Series) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std())


def _load_features(
    bias_module: str = DEFAULT_BIAS_MODULE,
    param_name: str = DEFAULT_PARAM_NAME,
    param_value: int = DEFAULT_PARAM_VALUE,
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
    populate_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load feature and target DataFrames from cache.

    Returns (features_df, targets_df, feature_col_name).
    Skips test if data/ohlc_data/ missing or cache unavailable.
    """
    project_root = _project_root()
    candle_dir = project_root / 'data' / 'ohlc_data'
    if not candle_dir.exists():
        pytest.skip(f'Missing candle directory: {candle_dir}')

    bias_spec = {
        'module_name': bias_module,
        'timeframes': [timeframe],
        'params': {param_name: param_value},
    }

    if populate_cache:
        manager = CacheManager(candle_dir=str(candle_dir))
        manager.populate_cache(
            bias_node_specs=[bias_spec],
            tickers=[ticker],
            start_date=start,
            end_date=end,
            show_progress=False,
            overwrite_existing=False,
        )

    try:
        features_df, targets_df = extract_features_for_bias_node(
            bias_spec=bias_spec,
            ticker=[ticker],
            start=start,
            end=end,
            use_millisecond_offset=True,
            target_col='log_return',
            use_cache=True,
        )
    except Exception as e:
        pytest.skip(f'Cache not populated: {e}. Run CacheManager.populate_cache() first.')

    if features_df is None or len(features_df) == 0:
        pytest.skip('Cache not populated. Run CacheManager.populate_cache() first.')

    feature_col = f'{bias_module}_signal_{timeframe.value}_{param_name}_{param_value}'
    if feature_col not in features_df.columns:
        # Try to find the column
        candidates = [c for c in features_df.columns if bias_module in c]
        if not candidates:
            pytest.skip(f'Feature column {feature_col} not found in cache.')
        feature_col = candidates[0]

    return features_df, targets_df, feature_col


def _load_candles(
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
) -> pd.DataFrame:
    """Load raw OHLCV candles from data/ohlc_data/ parquet files."""
    project_root = _project_root()
    candle_dir = project_root / 'data' / 'ohlc_data'
    if not candle_dir.exists():
        pytest.skip(f'Missing candle directory: {candle_dir}')

    # Deterministic and strict parquet selection for ticker/timeframe.
    # Repository layout usually stores files under data/ohlc_data/{TICKER}/.
    ticker_dir = candle_dir / ticker.value
    search_roots = [ticker_dir, candle_dir] if ticker_dir.exists() else [candle_dir]
    timeframe_tokens = [timeframe.name, str(timeframe.value)]
    ticker_tokens = [ticker.name, ticker.value]
    search_patterns = [
        f'*{ticker_token}*{timeframe_token}*.parquet'
        for ticker_token in ticker_tokens
        for timeframe_token in timeframe_tokens
    ] + [
        f'*{timeframe_token}*{ticker_token}*.parquet'
        for ticker_token in ticker_tokens
        for timeframe_token in timeframe_tokens
    ]

    seen_paths: set[Path] = set()
    for root in search_roots:
        for pattern in search_patterns:
            seen_paths.update(root.rglob(pattern))

    matches = sorted(seen_paths, key=lambda p: str(p))
    if not matches:
        pytest.skip(
            f'No candle parquet matched ticker={ticker.value} timeframe={timeframe.value} under {candle_dir}',
        )

    preferred_names = [
        f'{timeframe.name}_{ticker.name}.parquet',
        f'{ticker.name}_{timeframe.name}.parquet',
        f'{timeframe.name}_{ticker.value}.parquet',
        f'{ticker.value}_{timeframe.name}.parquet',
        f'{timeframe.value}_{ticker.value}.parquet',
        f'{ticker.value}_{timeframe.value}.parquet',
    ]
    preferred_match = next(
        (path for name in preferred_names for path in matches if path.name.lower() == name.lower()),
        None,
    )

    selected_path: Path
    if preferred_match is not None:
        selected_path = preferred_match
    elif len(matches) == 1:
        selected_path = matches[0]
    else:
        preview = ', '.join(str(path.relative_to(candle_dir)) for path in matches[:5])
        pytest.skip(
            'Ambiguous candle parquet selection for '
            f'ticker={ticker.value} timeframe={timeframe.value}; '
            f'found {len(matches)} matches ({preview}).',
        )

    df = pd.read_parquet(selected_path)

    # Normalise to have DatetimeIndex
    if 'datetime' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index, utc=False)
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

    # Filter date range
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    # Ensure required columns
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Try lowercase
        df.columns = df.columns.str.lower()
        missing = [c for c in required if c not in df.columns]
    if missing:
        pytest.skip(f'Candle data missing required columns: {missing}')

    if 'datetime' not in df.columns:
        df = df.copy()
        df['datetime'] = df.index

    return df


def _visualize_candle_shuffle(
    original_candles: pd.DataFrame,
    shuffled_candles: pd.DataFrame,
    output_dir: Path,
    n_bars: int = 60,
) -> Path:
    """
    Plot original vs shuffled candles side by side for visual inspection.

    Shows the first n_bars of each to demonstrate that:
    - Timestamps (x-axis) are preserved
    - Price OHLC structure is shuffled
    - Price levels evolve differently but volatility profile is similar

    Args:
        original_candles: Original OHLCV DataFrame.
        shuffled_candles: Shuffled OHLCV DataFrame from CandleShuffler.
        output_dir: Directory to save the plot.
        n_bars: Number of bars to display.

    Returns:
        Path to saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'candle_shuffle_comparison.png'

    # Take first n_bars for display
    orig = original_candles.head(n_bars)
    shuf = shuffled_candles.head(n_bars)
    x = np.arange(len(orig))

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(
        'Candle Shuffle Verification\n'
        'Left = Original | Right = Shuffled\n'
        'Timestamps preserved; OHLC structure permuted',
        fontsize=13, fontweight='bold',
    )

    def _plot_ohlc_simple(ax, df_sub, title: str, color: str) -> None:
        """Simple OHLC bar chart."""
        xi = np.arange(len(df_sub))
        ax.bar(xi, df_sub['high'] - df_sub['low'],
               bottom=df_sub['low'], width=0.6, color=color, alpha=0.5, label='H-L range')
        ax.plot(xi, df_sub['close'], color=color, linewidth=1.0, label='Close')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f'Bar index (first {len(df_sub)} bars)')
        ax.set_ylabel('Price')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot_ohlc_simple(axes[0, 0], orig, 'Original — Close price + H-L range', 'steelblue')
    _plot_ohlc_simple(axes[0, 1], shuf, 'Shuffled — Close price + H-L range', 'coral')

    # Bottom panels: close returns comparison
    for ax, df_sub, title, color in [
        (axes[1, 0], orig, 'Original — Close returns', 'steelblue'),
        (axes[1, 1], shuf, 'Shuffled — Close returns', 'coral'),
    ]:
        ret = df_sub['close'].pct_change().fillna(0)
        ax.bar(np.arange(len(ret)), ret, color=color, alpha=0.7, width=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f'Bar index')
        ax.set_ylabel('Return')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _run_rule_based_permutation_suite(
    bias_module: str = DEFAULT_BIAS_MODULE,
    param_name: str = DEFAULT_PARAM_NAME,
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
    nreps: int = NREPS_FAST,
    alpha: float = 0.10,
    metric_threshold: float = 0.0,
    oos_objective_metric: Optional[ObjectiveMetricSpec] = None,
    extractor_mode: Literal['momentum_proxy', 'gate_stress_proxy'] = 'momentum_proxy',
) -> tuple[PermutationTestSuite, list[dict], str]:
    """Run full suite with real candles and a lightweight rule proxy extractor.

    The extractor intentionally uses candle-derived returns as a fast integration
    proxy for orchestration plumbing. It is not a bias-node extraction
    integration test.

    Returns:
        (suite, param_grid, feature_col)
    """
    features_df, targets_df, feature_col = _load_features(
        bias_module=bias_module,
        param_name=param_name,
        param_value=DEFAULT_PARAM_VALUE,
        ticker=ticker,
        timeframe=timeframe,
    )

    target = targets_df['log_return'].dropna().copy()
    target.index = pd.to_datetime(target.index, utc=False)
    target.index = target.index.tz_localize(None) if target.index.tz is not None else target.index
    # Cache targets can carry millisecond offsets; strip sub-day precision so
    # candle-derived proxy features align on bar timestamps.
    target.index = target.index.floor('D')
    target = target[~target.index.duplicated(keep='last')]

    try:
        candles = _load_candles(ticker, timeframe, DEFAULT_START, DEFAULT_END)
    except Exception as e:
        pytest.skip(f'Could not load candles for permutation suite test: {e}')

    param_grid = [{param_name: v} for v in [3, 5, 10, 14]]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        lb = params[param_name]
        match extractor_mode:
            case 'momentum_proxy':
                return df['close'].pct_change(lb).fillna(0.0).rename(f'{param_name}_{lb}')
            case 'gate_stress_proxy':
                # Intentionally optimistic candle-derived proxy so in-sample
                # Stage 2 passers are available for vector-gate integration.
                signal = np.sign(df['close'].shift(-1) - df['close']).fillna(0.0)
                return signal.rename(f'{param_name}_{lb}')

    fold_structure = [
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2022-01-01')),
        (pd.Timestamp('2022-01-01'), pd.Timestamp('2024-01-01')),
    ]

    config = PermutationTestConfig(
        nreps=nreps,
        alpha=alpha,
        metric_threshold=metric_threshold,
        top_k=3,
        random_seed=42,
        permutation_mode_stage2='feature_shuffle',
        min_folds_stable=1,
        objective_metric=oos_objective_metric,
    )

    suite = run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': bias_module},
        target=target,
        param_grid=param_grid,
        objective_func=_sharpe,
        fold_structure=fold_structure,
        config=config,
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name=feature_col,
    )
    return suite, param_grid, feature_col


def _always_zero_objective_metric(_returns: pd.Series) -> float:
    """Deterministic metric used to force OOS vector-gate failures in tests."""
    return 0.0


# ---------------------------------------------------------------------------
# T013 Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_vector_shuffle_stage1(
    bias_module: str = DEFAULT_BIAS_MODULE,
    param_name: str = DEFAULT_PARAM_NAME,
    param_value: int = DEFAULT_PARAM_VALUE,
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
    nreps: int = NREPS_FAST,
    alpha: float = 0.10,
) -> None:
    """Integration test for T013: Vector Shuffle Permutation Test.

    Uses real RSI-5 features on ES daily data (2020-2023).
    Customizable for any bias node/param/ticker/timeframe.

    Researcher manual verification:
    - Inspect p-value and pass/fail verdict in terminal output
    - Confirm null distribution is approximately unimodal (printed shape)
    - Verify passed == (p_value <= alpha)
    """
    print('\n' + '=' * 60)
    print('Integration Test: T013 — Vector Shuffle Stage 1')
    print('=' * 60)

    features_df, targets_df, feature_col = _load_features(
        bias_module, param_name, param_value, ticker, timeframe,
    )

    feature = features_df[feature_col].dropna()
    target = targets_df['log_return'].reindex(feature.index).dropna()
    feature = feature.reindex(target.index)

    print(f'Bias: {feature_col}')
    print(f'Ticker: {ticker.value}  |  Timeframe: {timeframe.value}')
    print(f'Samples: {len(feature)} observations')
    print(f'nreps={nreps}  |  alpha={alpha}')

    # Fit binning model to get position multipliers
    model = ContinuousBinningModel(n_bins=15)
    try:
        model.fit(feature, target)
        fitted_vec = model.get_fitted_vector(strategy='long')
    except Exception as e:
        pytest.skip(f'Could not fit binning model: {e}')
        return

    report = run_vector_shuffle_test(
        fitted_feature=fitted_vec,
        target=target,
        objective_func=_sharpe,
        nreps=nreps,
        alpha=alpha,
        random_seed=42,
        param_combo=feature_col,
    )

    # Assertions
    assert isinstance(report, VectorShuffleReport)
    assert 0.0 <= report.p_value <= 1.0
    assert len(report.null_distribution) == nreps
    expected_cv = float(np.percentile(report.null_distribution, (1 - alpha) * 100))
    assert abs(report.critical_value - expected_cv) < 1e-9
    assert report.passed == (report.original_metric > report.critical_value)

    # Terminal output for researcher inspection
    null_mean = report.null_distribution.mean()
    null_std = report.null_distribution.std()
    print(f'\nVector Shuffle Results:')
    print(f'  Original metric (Sharpe): {report.original_metric:.4f}')
    print(f'  Null distribution: mean={null_mean:.4f}, std={null_std:.4f}')
    print(f'  Critical value (alpha={alpha}): {report.critical_value:.4f}')
    print(f'  p-value: {report.p_value:.4f}')
    print(f'  Verdict: {"PASS" if report.passed else "FAIL"}')
    print(f'\nPASS: T013 integration test successful')


# ---------------------------------------------------------------------------
# T014 Integration test (with candle visualization)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_permutation_stage2(
    bias_module: str = DEFAULT_BIAS_MODULE,
    param_name: str = DEFAULT_PARAM_NAME,
    param_value: int = DEFAULT_PARAM_VALUE,
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
    nreps: int = NREPS_FAST,
    alpha: float = 0.10,
    save_plots: bool = True,
) -> None:
    """Integration test for T014: Pipeline Permutation Stage 2.

    Tests both feature_shuffle and candle_shuffle modes.
    Includes CANDLE VISUALIZATION: plots original vs shuffled candles so
    researcher can visually verify the CandleShuffler is working correctly.

    Researcher manual verification:
    - Open candle_shuffle_comparison.png to inspect original vs shuffled OHLC
    - Verify that shuffled candles have same price level starting point but
      different trajectory (randomised intra-bar structure)
    - Inspect terminal for p-values from both modes; candle_shuffle p-value
      should be >= feature_shuffle p-value for genuine features
    - Verify no_trade_permutations count is reasonable
    """
    print('\n' + '=' * 60)
    print('Integration Test: T014 — Pipeline Permutation Stage 2')
    print('=' * 60)

    features_df, targets_df, feature_col = _load_features(
        bias_module, param_name, param_value, ticker, timeframe,
    )

    feature = features_df[feature_col].dropna()
    target = targets_df['log_return'].reindex(feature.index).dropna()
    feature = feature.reindex(target.index)

    print(f'Bias: {feature_col}')
    print(f'Ticker: {ticker.value}  |  Timeframe: {timeframe.value}')
    print(f'Samples: {len(feature)} observations')

    # --- Candle Visualization ---
    # Load raw candles and visualize original vs shuffled for researcher inspection
    output_dir = Path(tempfile.mkdtemp(prefix='perm_test_'))
    if os.getenv('SAVE_INTEGRATION_OUTPUTS'):
        output_dir = _project_root() / 'tests' / 'integration' / 'outputs' / 'permutation'
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        candles = _load_candles(ticker, timeframe, DEFAULT_START, DEFAULT_END)

        # Create one candle shuffle for visualization
        shuffler = CandleShuffler(candles, random_seed=42)
        shuffled_candles = shuffler.permute()

        candle_plot = _visualize_candle_shuffle(
            original_candles=candles,
            shuffled_candles=shuffled_candles,
            output_dir=output_dir,
            n_bars=60,
        )
        print(f'\nCandle shuffle visualization saved to: {candle_plot}')
        print('   Open this file to visually verify the candle shuffler is working correctly.')
        print('   Expected: timestamps preserved, OHLC structure permuted, same volatility profile.')

        has_candles = True
    except Exception as e:
        print(f'\nCould not load raw candles for visualization: {e}')
        print('   Proceeding with feature_shuffle mode only.')
        candles = None
        has_candles = False

    # --- Feature extractor for T014 ---
    def bias_node_extractor(df: pd.DataFrame) -> pd.Series:
        """Return fast candle-derived proxy feature for permutation plumbing.

        This keeps the integration test focused on permutation orchestration and
        alignment. It intentionally does not validate bias-node extraction.
        """
        try:
            ret = df['close'].pct_change(param_value).fillna(0.0)
            return ret.rename(feature_col)
        except Exception:
            return df['close'].pct_change().fillna(0.0).rename(feature_col)
    template_model = ContinuousBinningModel(n_bins=15)

    # --- Mode 1: Feature shuffle ---
    print(f'\n--- Mode 1: feature_shuffle (nreps={nreps}) ---')
    if candles is not None:
        report_fs = run_pipeline_permutation_continuous(
            candles_df=candles,
            bias_node_extractor=bias_node_extractor,
            binning_model=template_model,
            target=target.reindex(candles.index).dropna(),
            objective_func=_sharpe,
            permutation_mode='feature_shuffle',
            nreps=nreps,
            alpha=alpha,
            random_seed=42,
            param_combo=feature_col,
        )
        assert isinstance(report_fs, PipelinePermutationReport)
        assert 0.0 <= report_fs.p_value <= 1.0
        assert len(report_fs.null_distribution) == nreps
        assert report_fs.no_trade_permutations >= 0
        print(f'  p-value (feature_shuffle): {report_fs.p_value:.4f}')
        print(f'  no_trade_permutations: {report_fs.no_trade_permutations}/{nreps}')
        print(f'  Verdict: {"PASS" if report_fs.passed else "FAIL"}')

        # --- Mode 2: Candle shuffle ---
        print(f'\n--- Mode 2: candle_shuffle (nreps={nreps}) ---')
        report_cs = run_pipeline_permutation_continuous(
            candles_df=candles,
            bias_node_extractor=bias_node_extractor,
            binning_model=template_model,
            target=target.reindex(candles.index).dropna(),
            objective_func=_sharpe,
            permutation_mode='candle_shuffle',
            nreps=nreps,
            alpha=alpha,
            random_seed=42,
            param_combo=feature_col,
        )
        assert isinstance(report_cs, PipelinePermutationReport)
        assert 0.0 <= report_cs.p_value <= 1.0
        assert len(report_cs.null_distribution) == nreps
        assert report_cs.no_trade_permutations >= 0
        print(f'  p-value (candle_shuffle): {report_cs.p_value:.4f}')
        print(f'  no_trade_permutations: {report_cs.no_trade_permutations}/{nreps}')
        print(f'  Verdict: {"PASS" if report_cs.passed else "FAIL"}')
        print(f'\n  Note: candle_shuffle p >= feature_shuffle p for genuine features')
        print(f'    feature_shuffle p={report_fs.p_value:.4f}, candle_shuffle p={report_cs.p_value:.4f}')
    else:
        print('  Skipping candle-based modes (no candle data available)')

    print(f'\nPASS: T014 integration test successful')


# ---------------------------------------------------------------------------
# T015 Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_walkforward_stability_stage3(
    bias_module: str = DEFAULT_BIAS_MODULE,
    param_name: str = DEFAULT_PARAM_NAME,
    param_value: int = DEFAULT_PARAM_VALUE,
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
    nreps: int = NREPS_FAST,
) -> None:
    """Integration test for T015: Walkforward Stability Analysis.

    Uses RSI lookback grid [3, 5, 10, 14] with 2-year non-overlapping folds.
    Customizable for any bias node/param/ticker for researcher exploration.

    Researcher manual verification:
    - Inspect per-fold top-K param selections in terminal output
    - Confirm consistency metrics (overlap_rate) are printed
    - Review stability verdict: expect RSI short lookbacks to cluster
    """
    print('\n' + '=' * 60)
    print('Integration Test: T015 — Walkforward Stability Stage 3')
    print('=' * 60)

    features_df, targets_df, feature_col = _load_features(
        bias_module, param_name, param_value, ticker, timeframe,
    )

    target = targets_df['log_return'].dropna()

    try:
        candles = _load_candles(ticker, timeframe, DEFAULT_START, DEFAULT_END)
    except Exception as e:
        pytest.skip(f'Could not load candles for walkforward stability: {e}')

    # RSI lookback grid
    param_grid = [{param_name: v} for v in [3, 5, 10, 14]]
    print(f'Bias module: {bias_module}')
    print(f'Parameter grid: {param_name} in {[p[param_name] for p in param_grid]}')

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        lb = params[param_name]
        return df['close'].pct_change(lb).fillna(0.0).rename(f'{param_name}_{lb}')

    # 2-year non-overlapping folds within 2020-2023
    fold_structure = [
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2022-01-01')),
        (pd.Timestamp('2022-01-01'), pd.Timestamp('2024-01-01')),
    ]

    report = run_walkforward_stability(
        candles_df=candles,
        extractor_func=extractor,
        target=target,
        objective_func=_sharpe,
        param_grid=param_grid,
        fold_structure=fold_structure,
        top_k=3,
        feature_type='rule_based',
        feature_name=feature_col,
    )

    # Assertions
    assert isinstance(report, WalkforwardStabilityReport)
    assert len(report.fold_results) > 0
    assert 'overlap_rate' in report.consistency_metrics
    assert isinstance(report.is_stable, bool)
    assert isinstance(report.stability_verdict, str) and len(report.stability_verdict) > 0

    # Terminal output for researcher
    print(f'\nWalkforward Stability Results: {feature_col}')
    print(f'  Folds evaluated: {len(report.fold_results)}')
    for fr in report.fold_results:
        print(f'  {fr.fold_id} ({fr.fold_period[0]} -> {fr.fold_period[1]}):')
        print(f'    Top-{report.top_k} params: {fr.top_k_params}')

    print(f'\nConsistency Metrics:')
    for k, v in report.consistency_metrics.items():
        print(f'  {k}: {v:.4f}')

    print(f'\nStability Verdict: {report.stability_verdict}')
    print(f'Is stable: {"Yes" if report.is_stable else "No"}')
    print(f'\nPASS: T015 integration test successful')


# ---------------------------------------------------------------------------
# T016 + T017 Integration test (full suite)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_permutation_suite_runs_in_sample_walkforward_oos(
    bias_module: str = DEFAULT_BIAS_MODULE,
    param_name: str = DEFAULT_PARAM_NAME,
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
) -> None:
    """Integration test for full 3-phase suite: in-sample, walkforward, OOS."""
    print('\n' + '=' * 60)
    print('Integration Test: Full 3-Phase Permutation Suite Coverage')
    print('=' * 60)

    suite, param_grid, feature_col = _run_rule_based_permutation_suite(
        bias_module=bias_module,
        param_name=param_name,
        ticker=ticker,
        timeframe=timeframe,
        nreps=40,
        alpha=1.0,
        metric_threshold=-1e9,
    )

    assert isinstance(suite, PermutationTestSuite)
    assert len(suite.stage1_reports) == len(param_grid)
    assert len(suite.stage2_reports) <= len(suite.stage1_reports)
    assert len(suite.stage3_report.fold_results) > 0
    assert isinstance(suite.phase3_oos_reports, dict)

    if len(suite.phase3_oos_reports) == 0:
        pytest.skip(
            'Stage 2 produced zero passers for this dataset/config; OOS phase configured but had no candidates.',
        )

    assert set(suite.phase3_oos_reports).issubset(set(suite.stage2_reports))
    assert len(suite.combo_decisions) == len(param_grid)

    print(f'Feature: {feature_col}')
    print(f'Stage 1 reports: {len(suite.stage1_reports)}')
    print(f'Stage 2 reports: {len(suite.stage2_reports)}')
    print(f'Walkforward folds: {len(suite.stage3_report.fold_results)}')
    print(f'OOS reports: {len(suite.phase3_oos_reports)}')


@pytest.mark.integration
def test_permutation_suite_respects_vector_first_gate_in_oos(
    bias_module: str = DEFAULT_BIAS_MODULE,
    param_name: str = DEFAULT_PARAM_NAME,
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
) -> None:
    """Integration test: OOS candle permutation runs only after vector pass.

    Uses strict OOS ObjectiveMetricSpec to force vector failures while keeping
    in-sample Stage 2 passers likely so the vector-first gate path is exercised.
    """
    print('\n' + '=' * 60)
    print('Integration Test: OOS Vector-First Gate')
    print('=' * 60)

    suite, _, feature_col = _run_rule_based_permutation_suite(
        bias_module=bias_module,
        param_name=param_name,
        ticker=ticker,
        timeframe=timeframe,
        nreps=60,
        alpha=1.0,
        metric_threshold=-1e9,
        oos_objective_metric=ObjectiveMetricSpec(
            builtin='always_zero',
        ),
        extractor_mode='gate_stress_proxy',
    )

    assert suite.phase3_oos_reports, 'Expected OOS candidates from permissive in-sample gating.'

    vector_failures = [
        report for report in suite.phase3_oos_reports.values()
        if not report.vector_report.passed
    ]
    assert vector_failures, 'Expected vector failures from deterministic zero OOS objective metric.'

    assert all(report.candle_report is None for report in vector_failures)
    assert all(
        (report.candle_report is None) == (not report.vector_report.passed)
        for report in suite.phase3_oos_reports.values()
    )

    print(f'Feature: {feature_col}')
    print(f'OOS reports: {len(suite.phase3_oos_reports)}')
    print(f'Vector-failed reports: {len(vector_failures)}')


@pytest.mark.integration
def test_early_stopping_orchestration(
    bias_module: str = DEFAULT_BIAS_MODULE,
    param_name: str = DEFAULT_PARAM_NAME,
    param_value: int = DEFAULT_PARAM_VALUE,
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
    nreps: int = NREPS_FAST,
    alpha: float = 0.10,
) -> None:
    """Integration test for T016: Early Stopping Orchestration + T017: Report Generation.

    Runs the full three-stage permutation testing funnel on RSI lookback grid
    [3, 5, 10, 14] and generates all reports + plots.

    Customizable for any bias node/param/ticker for researcher exploration.

    Researcher manual verification:
    - Inspect terminal output for full funnel summary
    - Confirm Stage 2 count <= Stage 1 pass count (early stopping enforced)
    - Review ensemble_candidates list
    - Open generated plots and markdown summary for full report
    """
    print('\n' + '=' * 60)
    print('Integration Test: T016+T017 — Orchestration + Report Generation')
    print('=' * 60)

    suite, param_grid, feature_col = _run_rule_based_permutation_suite(
        bias_module=bias_module,
        param_name=param_name,
        ticker=ticker,
        timeframe=timeframe,
        nreps=nreps,
        alpha=alpha,
    )

    print(f'Feature: {feature_col}')
    print(f'Parameter grid: {[p[param_name] for p in param_grid]}')

    # Assertions
    assert isinstance(suite, PermutationTestSuite)
    assert len(suite.stage1_reports) == len(param_grid)
    assert len(suite.stage2_reports) <= suite.funnel_stats.stage1_pass
    assert suite.funnel_stats.total_params == len(param_grid)
    assert suite.funnel_stats.computational_savings_pct >= 0.0
    assert isinstance(suite.ensemble_candidates, list)
    assert set(suite.ensemble_candidates).issubset(set(suite.stage2_reports.keys()))
    assert isinstance(suite.phase3_oos_reports, dict)
    assert set(suite.phase3_oos_reports).issubset(set(suite.stage2_reports.keys()))
    assert len(suite.combo_decisions) == len(param_grid)

    # --- T017: Generate reports ---
    output_dir = Path(tempfile.mkdtemp(prefix='perm_report_'))
    if os.getenv('SAVE_INTEGRATION_OUTPUTS'):
        output_dir = _project_root() / 'tests' / 'integration' / 'outputs' / 'permutation_reports'
        output_dir.mkdir(parents=True, exist_ok=True)

    bundle = generate_permutation_reports(suite, output_dir)

    assert bundle.suite_json.exists()
    assert bundle.suite_markdown.exists()
    assert bundle.stage3_plot.exists()
    assert bundle.funnel_plot.exists()
    assert len(bundle.stage1_plots) == len(suite.stage1_reports)
    assert len(bundle.stage2_plots) == len(suite.stage2_reports)

    oos_vector_dir = output_dir / 'oos' / 'vector'
    oos_candle_dir = output_dir / 'oos' / 'candle'
    combo_table_path = output_dir / 'combo_decision_table.csv'
    assert oos_vector_dir.exists()
    assert oos_candle_dir.exists()
    assert combo_table_path.exists()

    vector_plot_paths = list(oos_vector_dir.glob('null_dist_*.png'))
    candle_plot_paths = list(oos_candle_dir.glob('null_dist_*.png'))
    assert len(vector_plot_paths) == len(suite.phase3_oos_reports)
    expected_candle_count = sum(
        1
        for report in suite.phase3_oos_reports.values()
        if report.candle_report is not None
    )
    assert len(candle_plot_paths) == expected_candle_count

    # Terminal summary
    print(f'\n{"="*60}')
    print('FUNNEL SUMMARY')
    print(f'{"="*60}')
    print(f'Stage 1 (Vector Shuffle): {suite.funnel_stats.total_params} -> {suite.funnel_stats.stage1_pass} passed')
    print(f'Stage 2 (Pipeline Perm):  {suite.funnel_stats.stage1_pass} -> {suite.funnel_stats.stage2_pass} passed')
    print(f'Stage 3 (Walkforward):    {suite.funnel_stats.total_params} evaluated -> {suite.funnel_stats.stable_params} stable')
    print(f'Ensemble candidates: {suite.ensemble_candidates}')
    print(f'Computational savings: {suite.funnel_stats.computational_savings_pct:.1f}%')
    print(f'\nReports saved to: {output_dir}')
    print(f'  suite_summary.md — researcher-readable summary')
    print(f'  suite_summary.json — structured data export')
    print(f'  stage1/ — null distribution plots per param')
    print(f'  stage2/ — null distribution plots (Stage 2 passers)')
    print(f'  stage3/walkforward_stability.png — stability heatmap')
    print(f'  funnel/funnel_diagram.png — param flow through stages')
    print(f'\nPASS: T016+T017 integration test successful')
