"""
Integration Test for Portfolio System

Tests the complete flow: candles -> ensembles -> portfolio -> position fractions
Uses real data (ES, NQ, YM, RTY) and hardcoded buy_hold feature config.

This is a true integration test that validates the entire pipeline works correctly.

Usage:
    # Run as script
    python tests/integration/test_portfolio_integration.py
    
    # Run with pytest
    pytest tests/integration/test_portfolio_integration.py -v
    
Test Scenarios:
    1. Refit scenario: Ensemble refitted from scratch on 2010-2020 data, tested on 2020-2024
    2. Pre-fit scenario: Ensemble uses pre-fitted params from config, tested on 2020-2024
    3. DataFrame format: Validates that all methods accept DataFrame inputs correctly

Note: This test uses hardcoded feature config to ensure it works even if vault files are deleted.
"""

import os

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

from utils.core.enums import TimeFrame, Ticker
from utils.core import helpers
from ensemble.portfolio import Portfolio
from ensemble.diversified_ensemble import DiversifiedEnsemble
from metrics.plotting.graphing.quantstats_reports import generate_tearsheet
from ensemble.portfolio_tester import (
    PortfolioTester,
    calculate_log_returns_from_candles,
    calculate_strategy_returns_from_positions,
    calculate_baseline_returns
)
from utils.compute.daily_ewsd_volatility import compute_daily_ewsd_volatility


# Hardcoded buy_hold feature config (from vault/D/buy_hold_long/features/buy_hold_signal_D.json)
# This ensures the test works even if the vault file is deleted
BUY_HOLD_FEATURE_CONFIG = {
    "feature_name": "buy_hold_signal_D",
    "feature_column": "buy_hold_signal_D",
    "created_at": "2026-01-21T03:47:49.085963+00:00",
    "updated_at": "2026-01-21T03:47:49.085987+00:00",
    "bias_node_spec": {
        "module_name": "buy_hold",
        "timeframes": ["D"],
        "params": {}
    },
    "tickers": ["ES", "NQ", "RTY", "YM"],
    "base_models": [
        {
            "model_id": "continuous_binning_2",
            "model_name": "buy_hold_signal_D::continuous_binning_2",
            "binning_model_type": "continuous_binning",
            "strategy": "long",
            "binning_model_params": {
                "n_bins": 2,
                "selection_metric": "sortino",
                "normalize_by": "ewsd"
            },
            "is_fitted": True,
            "fitted_params": {
                "model_version": "binning_v2",
                "bin_edges": [0.5],
                "bin_stats": {
                    "0": {
                        "mean_return": 0.0004376395916674674,
                        "std_return": 0.012884616482648844,
                        "downside_std": 0.009966707813405104,
                        "sortino_metric": 0.6970519524234261,
                        "sortino_metric_short": -0.7434033996590901,
                        "count": 1260,
                        "feature_min": 1.0,
                        "feature_max": 1.0
                    }
                },
                "active_bins_by_strategy": {
                    "long": [0],
                    "short": [],
                    "long_short": [0]
                },
                "position_multipliers_by_strategy": {
                    "long": {"0": 1.0},
                    "short": {},
                    "long_short": {"0": 1.0}
                }
            },
            "fitted_at": "2026-01-21T03:47:49.085981+00:00"
        }
    ]
}


def _daily_volatility_for(candles_df: pd.DataFrame) -> pd.DataFrame:
    if "timeframe" in candles_df.columns:
        daily_candles = candles_df[candles_df["timeframe"] == TimeFrame.D]
        if not daily_candles.empty:
            return compute_daily_ewsd_volatility(daily_candles)
    return compute_daily_ewsd_volatility(candles_df)


def create_ensemble_from_config(
    feature_config: Dict[str, Any],
    refit: bool = False
) -> DiversifiedEnsemble:
    """
    Create a DiversifiedEnsemble from hardcoded feature config.
    
    Parameters
    ----------
    feature_config : Dict[str, Any]
        Feature configuration (hardcoded BUY_HOLD_FEATURE_CONFIG)
    refit : bool, default=False
        If True, ensemble will be refitted from scratch.
        If False, uses fitted params from config.
        
    Returns
    -------
    DiversifiedEnsemble
        Configured ensemble instance
    """
    # Create a temporary control file structure
    # The ensemble expects a control file path, so we'll create a minimal one
    import tempfile
    import json
    
    # Convert base_models to the format expected by control file
    # Keep timeframes as strings in JSON (will be converted to TimeFrame enums when loading)
    base_models_config = []
    for model in feature_config['base_models']:
        base_model_config = {
            'name': model['model_name'],
            'feature_column': feature_config['feature_column'],
            'model_type': model['binning_model_type'],  # Use model_type, not binning_model_type
            'strategy': model['strategy'],
            'constructor_params': model['binning_model_params'],  # Use constructor_params, not binning_model_params
            'bias_node_spec': feature_config['bias_node_spec'],  # Include bias_node_spec (timeframes as strings for JSON)
            'members': [
                {'member_name': model['model_name'], 'params': model['binning_model_params']}
            ]
        }
        base_models_config.append(base_model_config)
    
    # Create temporary control file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        control_file = {
            'metadata': {
                'created_at': feature_config['created_at'],
                'updated_at': feature_config['updated_at'],
                'is_fit': not refit,  # If refit=False, mark as fitted
                'base_tf': 'D',
                **({'selection_method': 'manual'} if not refit else {})
            },
            'base_models': base_models_config,
            'tickers': feature_config['tickers']
        }
        
        # Only add fitted params if not refitting
        if not refit:
            control_file['fitted_base_models'] = {
                model['model_name']: model['fitted_params']
                for model in feature_config['base_models']
            }
            control_file['fitted_ensemble'] = {
                'weights': {model['model_name']: 1.0 for model in feature_config['base_models']},
                'exposure_fractions': {model['model_name']: 0.5 for model in feature_config['base_models']},
                'model_exposure_fractions': {model['model_name']: 0.5 for model in feature_config['base_models']},
                'feature_names': [model['model_name'] for model in feature_config['base_models']],
                'target_volatility': 0.20,
                'unique_tickers': feature_config['tickers'],
                'instrument_weights': {t: 0.25 for t in feature_config['tickers']},
                'n_tickers': len(feature_config['tickers'])
            }
        json.dump(control_file, f, indent=2)
        temp_path = f.name
    
    # Create ensemble from control file
    ensemble = DiversifiedEnsemble(
        control_file_path=temp_path,
        target_volatility=0.20,
        base_tf=TimeFrame.D,
        use_cache=False
    )
    
    # Clean up temp file after ensemble loads it
    # (ensemble loads it in __init__, so safe to delete now)
    os.unlink(temp_path)
    
    return ensemble


# Helper functions are now imported from ensemble.portfolio_tester
# Aliases for backward compatibility in test file
calculate_returns_from_candles = calculate_log_returns_from_candles
calculate_strategy_returns = calculate_strategy_returns_from_positions


class TestPortfolioIntegration:
    """Integration tests for Portfolio system with real data."""
    
    def test_portfolio_refit_scenario(self):
        """
        Test portfolio with pre-fitted ensemble on train/test split.

        Scenario:
        - Load real data (ES, NQ, YM, RTY) from 2010-2020 for training
        - Use pre-fitted params (buy_hold is constant, cannot refit with continuous_binning)
        - Test on 2020-2024 data
        - Validate outputs are position fractions
        """
        print("\n" + "="*70)
        print("TEST: Portfolio with Pre-Fitted Ensemble (train/test split)")
        print("="*70)
        
        # Load training data (2010-2020)
        print("\nLoading training data (2010-2020)...")
        train_candles = helpers.load_data_multi_ticker(
            tickers=[Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY],
            timeframe=TimeFrame.D,
            start=datetime(2010, 1, 1),
            end=datetime(2020, 1, 1),
        )
        print(f"Loaded {len(train_candles)} training candles")
        print(f"Tickers: {train_candles['ticker'].unique()}")
        
        # Load test data (2020-2024)
        print("\nLoading test data (2020-2024)...")
        test_candles = helpers.load_data_multi_ticker(
            tickers=[Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY],
            timeframe=TimeFrame.D,
            start=datetime(2020, 1, 1),
            end=datetime(2024, 1, 1),
        )
        print(f"Loaded {len(test_candles)} test candles")

        # Create ensemble with pre-fitted params (buy_hold is a constant feature
        # that cannot be refitted with continuous_binning)
        print("\nCreating ensemble (refit=False, pre-fitted)...")
        ensemble = create_ensemble_from_config(BUY_HOLD_FEATURE_CONFIG, refit=False)
        
        # Create portfolio
        print("\nCreating portfolio...")
        portfolio = Portfolio(
            ensembles=[ensemble],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            max_position_pct=2.0
        )
        
        # Create PortfolioTester
        tester = PortfolioTester(portfolio, baseline_mode='equal_weight')
        
        # Fit portfolio (automatically calculates returns)
        print("\nFitting portfolio...")
        tester.fit(train_candles)
        print(f"Portfolio fitted: {portfolio.is_fitted_}")
        print(f"IDM: {portfolio.idm_}")
        
        # Predict on test data
        print("\nPredicting on test data...")
        daily_volatility_df = _daily_volatility_for(pd.concat([train_candles, test_candles], ignore_index=True))
        positions = tester.predict(test_candles, daily_volatility_df=daily_volatility_df)
        
        # Validate outputs
        print("\nValidating outputs...")
        assert isinstance(positions, pd.DataFrame), "Output should be DataFrame"
        assert len(positions) > 0, "Should have predictions"
        
        required_cols = ['ticker', 'datetime', 'forecast_score', 'position_fraction']
        for col in required_cols:
            assert col in positions.columns, f"Missing column: {col}"
        
        # Validate position fractions are reasonable
        assert positions['position_fraction'].notna().all(), "No NaN position fractions"
        assert (positions['position_fraction'].abs() <= 2.0).all(), "Position fractions should be capped at 2.0"
        
        # CRITICAL: Validate that we actually have non-zero predictions
        non_zero_positions = (positions['position_fraction'] != 0).sum()
        non_zero_forecasts = (positions['forecast_score'] != 0).sum()
        
        assert non_zero_positions > 0, (
            f"CRITICAL: All position fractions are zero! "
            f"This indicates the ensemble/base models are not generating predictions. "
            f"Total positions: {len(positions)}, Non-zero: {non_zero_positions}"
        )
        assert non_zero_forecasts > 0, (
            f"CRITICAL: All forecast scores are zero! "
            f"This indicates the ensemble is not generating forecasts. "
            f"Total positions: {len(positions)}, Non-zero forecasts: {non_zero_forecasts}"
        )
        
        print(f"\n✓ Generated {len(positions)} position predictions")
        print(f"  Position fraction range: [{positions['position_fraction'].min():.4f}, {positions['position_fraction'].max():.4f}]")
        print(f"  Forecast score range: [{positions['forecast_score'].min():.4f}, {positions['forecast_score'].max():.4f}]")
        print(f"  Non-zero positions: {non_zero_positions} / {len(positions)} ({100*non_zero_positions/len(positions):.1f}%)")
        print(f"  Non-zero forecasts: {non_zero_forecasts} / {len(positions)} ({100*non_zero_forecasts/len(positions):.1f}%)")
        print(f"  Unique tickers: {sorted(positions['ticker'].unique())}")
        
        return positions
    
    def test_portfolio_use_fitted_params_scenario(self):
        """
        Test portfolio using pre-fitted ensemble parameters.
        
        Scenario:
        - Load real data (ES, NQ, YM, RTY) from 2010-2020 for training
        - Use pre-fitted params from config (don't refit)
        - Test on 2020-2024 data
        - Validate outputs are position fractions
        """
        print("\n" + "="*70)
        print("TEST: Portfolio with Pre-Fitted Ensemble")
        print("="*70)
        
        # Load training data (2010-2020)
        print("\nLoading training data (2010-2020)...")
        train_candles = helpers.load_data_multi_ticker(
            tickers=[Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY],
            timeframe=TimeFrame.D,
            start=datetime(2010, 1, 1),
            end=datetime(2020, 1, 1),
        )
        print(f"Loaded {len(train_candles)} training candles")
        
        # Load test data (2020-2024)
        print("\nLoading test data (2020-2024)...")
        test_candles = helpers.load_data_multi_ticker(
            tickers=[Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY],
            timeframe=TimeFrame.D,
            start=datetime(2020, 1, 1),
            end=datetime(2024, 1, 1),
        )
        print(f"Loaded {len(test_candles)} test candles")
        
        # Create ensemble (use pre-fitted params)
        print("\nCreating ensemble (refit=False)...")
        ensemble = create_ensemble_from_config(BUY_HOLD_FEATURE_CONFIG, refit=False)
        print(f"Ensemble fitted: {ensemble.is_fitted_}")
        
        # Create portfolio
        print("\nCreating portfolio...")
        portfolio = Portfolio(
            ensembles=[ensemble],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            max_position_pct=2.0
        )
        
        # Create PortfolioTester
        tester = PortfolioTester(portfolio, baseline_mode='equal_weight')
        
        # Ensemble should already be fitted from config (refit=False)
        print(f"\nEnsemble fitted status: {ensemble.is_fitted_}")
        
        # Fit portfolio (fits IDM, ensemble should already be fitted)
        print("\nFitting portfolio (IDM calculation)...")
        tester.fit(train_candles)
        print(f"Portfolio fitted: {portfolio.is_fitted_}")
        print(f"IDM: {portfolio.idm_}")
        
        # Predict on test data
        print("\nPredicting on test data...")
        daily_volatility_df = _daily_volatility_for(pd.concat([train_candles, test_candles], ignore_index=True))
        positions = tester.predict(test_candles, daily_volatility_df=daily_volatility_df)
        
        # Validate outputs
        print("\nValidating outputs...")
        assert isinstance(positions, pd.DataFrame), "Output should be DataFrame"
        assert len(positions) > 0, "Should have predictions"
        
        required_cols = ['ticker', 'datetime', 'forecast_score', 'position_fraction']
        for col in required_cols:
            assert col in positions.columns, f"Missing column: {col}"
        
        # Validate position fractions are reasonable
        assert positions['position_fraction'].notna().all(), "No NaN position fractions"
        assert (positions['position_fraction'].abs() <= 2.0).all(), "Position fractions should be capped at 2.0"
        
        # CRITICAL: Validate that we actually have non-zero predictions
        non_zero_positions = (positions['position_fraction'] != 0).sum()
        non_zero_forecasts = (positions['forecast_score'] != 0).sum()
        
        assert non_zero_positions > 0, (
            f"CRITICAL: All position fractions are zero! "
            f"This indicates the ensemble/base models are not generating predictions. "
            f"Total positions: {len(positions)}, Non-zero: {non_zero_positions}"
        )
        assert non_zero_forecasts > 0, (
            f"CRITICAL: All forecast scores are zero! "
            f"This indicates the ensemble is not generating forecasts. "
            f"Total positions: {len(positions)}, Non-zero forecasts: {non_zero_forecasts}"
        )
        
        print(f"\n✓ Generated {len(positions)} position predictions")
        print(f"  Position fraction range: [{positions['position_fraction'].min():.4f}, {positions['position_fraction'].max():.4f}]")
        print(f"  Forecast score range: [{positions['forecast_score'].min():.4f}, {positions['forecast_score'].max():.4f}]")
        print(f"  Non-zero positions: {non_zero_positions} / {len(positions)} ({100*non_zero_positions/len(positions):.1f}%)")
        print(f"  Non-zero forecasts: {non_zero_forecasts} / {len(positions)} ({100*non_zero_forecasts/len(positions):.1f}%)")
        print(f"  Unique tickers: {sorted(positions['ticker'].unique())}")
        
        return positions
    
    def test_portfolio_dataframe_format(self):
        """
        Test that portfolio methods accept DataFrame format correctly.
        
        Validates:
        - fit_from_candles() accepts candles DataFrame
        - predict_from_candles() accepts candles DataFrame
        - Output has correct format
        """
        print("\n" + "="*70)
        print("TEST: Portfolio DataFrame Format")
        print("="*70)
        
        # Load small sample of data
        print("\nLoading sample data...")
        candles = helpers.load_data_multi_ticker(
            tickers=[Ticker.ES, Ticker.NQ],
            timeframe=TimeFrame.D,
            start=datetime(2020, 1, 1),
            end=datetime(2020, 3, 1),  # Just 2 months
        )
        print(f"Loaded {len(candles)} candles")
        
        # Validate input format
        required_input_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'timeframe']
        for col in required_input_cols:
            assert col in candles.columns, f"Missing input column: {col}"
        
        # Create ensemble and portfolio
        ensemble = create_ensemble_from_config(BUY_HOLD_FEATURE_CONFIG, refit=False)
        portfolio = Portfolio(
            ensembles=[ensemble],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
        )
        
        # Create PortfolioTester
        tester = PortfolioTester(portfolio, baseline_mode='equal_weight')
        
        # Test fit accepts DataFrame
        print("\nTesting fit()...")
        tester.fit(candles)
        assert portfolio.is_fitted_, "Portfolio should be fitted"
        
        # Test predict accepts DataFrame
        print("\nTesting predict()...")
        daily_volatility_df = _daily_volatility_for(candles)
        positions = tester.predict(candles, daily_volatility_df=daily_volatility_df)
        
        # Validate output format
        assert isinstance(positions, pd.DataFrame), "Output should be DataFrame"
        assert 'ticker' in positions.columns, "Output should have ticker column"
        assert 'datetime' in positions.columns, "Output should have datetime column"
        assert 'position_fraction' in positions.columns, "Output should have position_fraction column"
        
        # Validate we have predictions
        assert len(positions) > 0, "Should have at least some predictions"
        
        # CRITICAL: Validate that we actually have non-zero predictions
        non_zero_positions = (positions['position_fraction'] != 0).sum()
        non_zero_forecasts = (positions['forecast_score'] != 0).sum() if 'forecast_score' in positions.columns else 0
        
        assert non_zero_positions > 0, (
            f"CRITICAL: All position fractions are zero! "
            f"Total positions: {len(positions)}, Non-zero: {non_zero_positions}"
        )
        
        print(f"\n✓ DataFrame format validated")
        print(f"  Total positions: {len(positions)}")
        print(f"  Non-zero positions: {non_zero_positions} / {len(positions)} ({100*non_zero_positions/len(positions):.1f}%)")
        return positions
    
    def test_base_model_predictions_non_empty(self):
        """
        Test that base models actually generate non-empty predictions.
        
        This is a critical test to catch bugs where base models return empty Series.
        """
        print("\n" + "="*70)
        print("TEST: Base Model Predictions Non-Empty")
        print("="*70)
        
        # Load small sample of data
        print("\nLoading sample data...")
        candles = helpers.load_data_multi_ticker(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2020, 1, 1),
            end=datetime(2020, 2, 1),  # Just 1 month
        )
        print(f"Loaded {len(candles)} candles")
        
        # Create ensemble
        ensemble = create_ensemble_from_config(BUY_HOLD_FEATURE_CONFIG, refit=False)
        assert ensemble.is_fitted_, "Ensemble should be fitted"
        
        # Get base model
        base_model_name = list(ensemble.base_models.keys())[0]
        base_model = ensemble.base_models[base_model_name]
        print(f"\nTesting base model: {base_model_name}")
        
        # Test base model prediction directly
        ticker_candles = candles[candles['ticker'] == Ticker.ES].copy()
        print(f"Testing with {len(ticker_candles)} candles for {Ticker.ES}")
        
        # Base model should return non-empty Series
        base_pred = base_model.predict(ticker_candles)
        print(f"Base model prediction type: {type(base_pred)}")
        print(f"Base model prediction length: {len(base_pred)}")
        print(f"Base model prediction range: [{base_pred.min():.4f}, {base_pred.max():.4f}]" if len(base_pred) > 0 else "Empty!")
        
        assert len(base_pred) > 0, (
            f"CRITICAL: Base model returned empty predictions! "
            f"This indicates the base model's predict() method is not working correctly. "
            f"Model: {base_model_name}, Input candles: {len(ticker_candles)}"
        )
        
        # Test ensemble prediction
        print("\nTesting ensemble prediction...")
        daily_volatility_df = _daily_volatility_for(candles)
        ensemble_pred = ensemble.predict_from_candles(
            candles,
            daily_volatility_df=daily_volatility_df,
        )
        print(f"Ensemble prediction type: {type(ensemble_pred)}")
        if isinstance(ensemble_pred, pd.DataFrame):
            print(f"Ensemble prediction shape: {ensemble_pred.shape}")
            print(f"Ensemble forecast_score range: [{ensemble_pred['forecast_score'].min():.4f}, {ensemble_pred['forecast_score'].max():.4f}]" if len(ensemble_pred) > 0 else "Empty!")
            
            assert len(ensemble_pred) > 0, (
                f"CRITICAL: Ensemble returned empty predictions! "
                f"This indicates the ensemble's predict_from_candles() method is not working correctly."
            )
            
            non_zero_forecasts = (ensemble_pred['forecast_score'] != 0).sum()
            assert non_zero_forecasts > 0, (
                f"CRITICAL: All ensemble forecast scores are zero! "
                f"Total: {len(ensemble_pred)}, Non-zero: {non_zero_forecasts}"
            )
        
        print("\n✓ Base model and ensemble predictions validated")
        return base_pred, ensemble_pred


def main():
    """Run integration tests."""
    print("="*70)
    print("PORTFOLIO INTEGRATION TESTS")
    print("="*70)
    
    test_suite = TestPortfolioIntegration()
    
    try:
        # Test 1: Refit scenario
        print("\n" + "="*70)
        print("Running Test 1: Portfolio with Refit Ensemble")
        print("="*70)
        positions_refit = test_suite.test_portfolio_refit_scenario()
        
        # Test 2: Use fitted params scenario
        print("\n" + "="*70)
        print("Running Test 2: Portfolio with Pre-Fitted Ensemble")
        print("="*70)
        positions_prefit = test_suite.test_portfolio_use_fitted_params_scenario()
        
        # Test 3: DataFrame format
        print("\n" + "="*70)
        print("Running Test 3: Portfolio DataFrame Format")
        print("="*70)
        positions_format = test_suite.test_portfolio_dataframe_format()
        
        # Test 4: Base model predictions non-empty
        print("\n" + "="*70)
        print("Running Test 4: Base Model Predictions Non-Empty")
        print("="*70)
        base_pred, ensemble_pred = test_suite.test_base_model_predictions_non_empty()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print(f"\nSummary:")
        print(f"  Refit scenario: {len(positions_refit)} positions")
        print(f"  Pre-fit scenario: {len(positions_prefit)} positions")
        print(f"  Format test: {len(positions_format)} positions")
        print(f"  Base model test: {len(base_pred)} base predictions, {len(ensemble_pred) if isinstance(ensemble_pred, pd.DataFrame) else 'N/A'} ensemble predictions")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
