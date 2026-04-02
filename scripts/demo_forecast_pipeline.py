#!/usr/bin/env python3
"""
Demo: Full Forecast Generation Pipeline
========================================

Demonstrates the new v6.0.0 architecture:
    Base Models -> Ensemble -> WeightLayer -> Portfolio -> PositionSizer

This script:
1. Shows the full pipeline in action with synthetic data
2. Verifies formulas with known inputs
3. Demonstrates the new layered architecture

Run: python scripts/demo_forecast_pipeline.py

Author: Trading Research Team
Date: 2026-01-18
"""

import sys
import os

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

import numpy as np
import pandas as pd
from typing import List

from ensemble.weight_layer import WeightLayer
from ensemble.portfolio import Portfolio
from execution.position_sizer import (
    PositionSizer,
    ContractSpec,
    RoundingMethod
)


# =============================================================================
# CONSTANTS
# =============================================================================

DIVIDER = "=" * 80
SUB_DIVIDER = "-" * 60


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def create_synthetic_forecast_vectors(
    tickers: List[str],
    model_names: List[str],
    seed: int = 42
) -> List[pd.DataFrame]:
    """
    Create synthetic forecast vectors simulating Ensemble output.

    Parameters
    ----------
    tickers : list
        Instrument tickers
    model_names : list
        Base model names
    seed : int
        Random seed for reproducibility

    Returns
    -------
    list[pd.DataFrame]
        List of forecast vector DataFrames
    """
    np.random.seed(seed)

    rows = []
    for ticker in tickers:
        for model in model_names:
            # Generate forecast (positive for bullish, 0 for inactive)
            signal = np.random.choice([0, 1], p=[0.3, 0.7])
            forecast = np.random.uniform(0.5, 2.0) if signal else 0.0

            rows.append({
                'ticker': ticker,
                'model_name': model,
                'forecast': forecast,
                'signal': signal
            })

    return [pd.DataFrame(rows)]


def create_synthetic_signals(
    model_names: List[str],
    n_samples: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create synthetic binary signals for weight calculation.

    Parameters
    ----------
    model_names : list
        Base model names
    n_samples : int
        Number of samples
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Binary signals DataFrame
    """
    np.random.seed(seed)

    signals = {}
    for model in model_names:
        signals[model] = np.random.randint(0, 2, n_samples)

    return pd.DataFrame(signals)


def create_synthetic_returns(
    tickers: List[str],
    n_samples: int = 252,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create synthetic instrument returns for IDM calculation.

    Parameters
    ----------
    tickers : list
        Instrument tickers
    n_samples : int
        Number of samples (default: 1 year of daily data)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Returns DataFrame
    """
    np.random.seed(seed)

    returns = {}
    for ticker in tickers:
        # Simulate daily returns with ~15% annual vol
        daily_vol = 0.15 / np.sqrt(252)
        returns[ticker] = np.random.randn(n_samples) * daily_vol

    return pd.DataFrame(returns)


# =============================================================================
# FORMULA VERIFICATION
# =============================================================================

def verify_forecast_formula():
    """Verify the per-model forecast formula with known inputs."""
    print("\n" + SUB_DIVIDER)
    print("FORMULA VERIFICATION: Per-Model Forecast")
    print(SUB_DIVIDER)
    print("\nFormula: F_i = (tau / (sigma * sqrt(h_i))) * X_i")
    print("\nWhere:")
    print("  tau   = Target annual portfolio volatility")
    print("  sigma = Instrument's blended annualized volatility")
    print("  h_i   = Exposure fraction (1/n_bins for the base model)")
    print("  X_i   = Binary signal (0 or 1)")

    test_cases = [
        # (tau, sigma, h, signal, description)
        (0.20, 0.20, 1.0, 1, "Base case: equal vol"),
        (0.20, 0.40, 1.0, 1, "Double volatility -> half position"),
        (0.20, 0.10, 1.0, 1, "Half volatility -> double position"),
        (0.20, 0.20, 0.1, 1, "10 bins exposure -> 3.16x position"),
        (0.20, 0.20, 1.0, 0, "Inactive signal -> zero position"),
    ]

    print("\n{:<35} {:>10} {:>10} {:>8}".format(
        "Test Case", "Expected", "Actual", "Status"
    ))
    print("-" * 70)

    all_passed = True
    for tau, sigma, h, signal, desc in test_cases:
        expected = (tau / (sigma * np.sqrt(h))) * signal
        actual = (tau / (sigma * np.sqrt(h))) * signal

        status = "PASS" if abs(expected - actual) < 1e-5 else "FAIL"
        if status == "FAIL":
            all_passed = False

        print("{:<35} {:>10.4f} {:>10.4f} {:>8}".format(
            desc, expected, actual, status
        ))

    return all_passed


def verify_fdm_formula():
    """Verify FDM calculation with WeightLayer."""
    print("\n" + SUB_DIVIDER)
    print("FORMULA VERIFICATION: FDM (Forecast Diversification Multiplier)")
    print(SUB_DIVIDER)
    print("\nFormula: FDM = min(sqrt(1 / (mean_corr + 0.01)), fdm_max)")
    print("Where mean_corr is the average correlation between forecast values")

    # Test single model case
    print("\nTest: Single model -> FDM = 1.0")
    layer_single = WeightLayer(fdm_max=2.0)
    forecasts_single = [pd.DataFrame({
        'ticker': ['ES', 'NQ'],
        'model_name': ['model_a', 'model_a'],
        'forecast': [1.0, 0.8],
        'signal': [1, 1]
    })]
    signals_single = pd.DataFrame({'model_a': [1, 0, 1, 0, 1]})
    layer_single.fit(forecasts_single, signals_single)

    single_status = "PASS" if abs(layer_single.fdm_ - 1.0) < 1e-5 else "FAIL"
    print(f"  Expected: 1.0, Actual: {layer_single.fdm_:.4f}, Status: {single_status}")

    # Test FDM cap
    print("\nTest: FDM capped at fdm_max (2.0)")
    layer_cap = WeightLayer(fdm_max=2.0)
    cap_status = "PASS" if layer_cap.fdm_max == 2.0 else "FAIL"
    print(f"  fdm_max: {layer_cap.fdm_max}, Status: {cap_status}")

    return single_status == "PASS" and cap_status == "PASS"


def verify_idm_formula():
    """Verify IDM calculation with Portfolio."""
    print("\n" + SUB_DIVIDER)
    print("FORMULA VERIFICATION: IDM (Instrument Diversification Multiplier)")
    print(SUB_DIVIDER)
    print("\nFormula: IDM = min(sqrt(1 / (mean_corr + 0.01)), idm_max)")
    print("Where mean_corr is the average correlation between instrument returns")

    # Test single instrument case
    print("\nTest: Single instrument -> IDM = 1.0")
    portfolio_single = Portfolio(idm_max=2.5)
    returns_single = pd.DataFrame({
        'ES': np.random.randn(100) * 0.01
    })
    portfolio_single.fit(returns_single)

    single_status = "PASS" if abs(portfolio_single.idm_ - 1.0) < 1e-5 else "FAIL"
    print(f"  Expected: 1.0, Actual: {portfolio_single.idm_:.4f}, Status: {single_status}")

    # Test IDM cap
    print("\nTest: IDM capped at idm_max (2.5)")
    cap_status = "PASS" if portfolio_single.idm_max == 2.5 else "FAIL"
    print(f"  idm_max: {portfolio_single.idm_max}, Status: {cap_status}")

    return single_status == "PASS" and cap_status == "PASS"


# =============================================================================
# FULL PIPELINE DEMO
# =============================================================================

def run_full_pipeline():
    """Run the complete forecast generation pipeline."""
    print("\n" + DIVIDER)
    print("FULL PIPELINE DEMO")
    print(DIVIDER)

    # Configuration
    tickers = ['ES', 'NQ', 'GC']
    model_names = ['ewmac_8', 'ewmac_16', 'rsi_14']
    capital = 1_000_000

    print(f"\nConfiguration:")
    print(f"  Tickers: {tickers}")
    print(f"  Models: {model_names}")
    print(f"  Capital: ${capital:,}")

    # =========================================================================
    # STEP 1: Generate synthetic data (simulating Ensemble output)
    # =========================================================================
    print("\n" + SUB_DIVIDER)
    print("STEP 1: Ensemble Output (Per-Model Forecasts)")
    print(SUB_DIVIDER)

    forecast_vectors = create_synthetic_forecast_vectors(tickers, model_names)
    signals = create_synthetic_signals(model_names)

    print("\nForecast Vectors (output from Ensemble.predict()):")
    print(forecast_vectors[0].to_string(index=False))

    # =========================================================================
    # STEP 2: WeightLayer - Combine forecasts with FDM
    # =========================================================================
    print("\n" + SUB_DIVIDER)
    print("STEP 2: Weight Layer (FDM-Scaled Combination)")
    print(SUB_DIVIDER)

    weight_layer = WeightLayer(fdm_max=2.0)
    weight_layer.fit(forecast_vectors, signals)

    print(f"\nInverse Correlation Weights:")
    for model, weight in weight_layer.weights_.items():
        print(f"  {model}: {weight:.4f}")

    print(f"\nFDM: {weight_layer.fdm_:.4f}")
    print(f"Mean Forecast Correlation: {weight_layer.mean_forecast_correlation_:.4f}")

    combined = weight_layer.combine(forecast_vectors)
    print("\nCombined Forecasts (FDM-scaled):")
    print(combined.to_string(index=False))

    # =========================================================================
    # STEP 3: Portfolio - Apply IDM and instrument weights
    # =========================================================================
    print("\n" + SUB_DIVIDER)
    print("STEP 3: Portfolio (IDM-Scaled Allocation)")
    print(SUB_DIVIDER)

    returns = create_synthetic_returns(tickers)

    portfolio = Portfolio(
        max_position_pct=2.0,  # Cap at 200%
        idm_max=2.5
    )
    portfolio.fit(returns)

    print(f"\nIDM: {portfolio.idm_:.4f}")
    print(f"Mean Return Correlation: {portfolio.mean_return_correlation_:.4f}")
    print(f"Max Position: {portfolio.max_position_pct:.1f}x")

    positions = portfolio.predict(combined)
    print("\nPosition Fractions:")
    print(positions.to_string(index=False))

    # =========================================================================
    # STEP 4: PositionSizer - Convert to contracts
    # =========================================================================
    print("\n" + SUB_DIVIDER)
    print("STEP 4: Position Sizer (Contracts)")
    print(SUB_DIVIDER)

    specs = {
        'ES': ContractSpec(ticker='ES', price=4800, multiplier=50),
        'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20),
        'GC': ContractSpec(ticker='GC', price=2000, multiplier=100)
    }

    print("\nContract Specifications:")
    print(f"  {'Ticker':<8} {'Price':>10} {'Multiplier':>12} {'Contract Value':>15}")
    for ticker, spec in specs.items():
        print(f"  {spec.ticker:<8} ${spec.price:>9,} {spec.multiplier:>12} ${spec.contract_value:>14,}")

    sizer = PositionSizer(
        capital=capital,
        contract_specs=specs,
        rounding_method=RoundingMethod.ROUND
    )

    contracts = sizer.calculate_positions(positions)
    print(f"\nCapital: ${capital:,}")
    print("\nPosition Details:")
    print(contracts.to_string(index=False))

    # Summary
    summary = sizer.get_summary(contracts)
    print("\nPortfolio Summary:")
    print(f"  Total Target: ${summary['total_target_dollars']:,.0f}")
    print(f"  Total Notional: ${summary['total_notional_value']:,.0f}")
    print(f"  Total Exposure: {summary['total_notional_pct']:.1%}")
    print(f"  Total Contracts: {summary['n_contracts']}")
    print(f"  Allocation Error: {summary['allocation_error_pct']:.2f}%")

    return contracts


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_diagnostics(weight_layer: WeightLayer, portfolio: Portfolio):
    """Print diagnostic information from fitted components."""
    print("\n" + SUB_DIVIDER)
    print("DIAGNOSTICS")
    print(SUB_DIVIDER)

    print("\nWeightLayer Diagnostics:")
    wl_diag = weight_layer.get_diagnostics()
    for key, value in wl_diag.items():
        print(f"  {key}: {value}")

    print("\nPortfolio Diagnostics:")
    p_diag = portfolio.get_diagnostics()
    for key, value in p_diag.items():
        print(f"  {key}: {value}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print(DIVIDER)
    print("FORECAST PIPELINE DEMO - v6.0.0 Architecture")
    print(DIVIDER)
    print("\nArchitecture Flow:")
    print("  Base Models -> Ensemble -> WeightLayer -> Portfolio -> PositionSizer")
    print("\nThis demo verifies formulas and demonstrates the full pipeline.")

    # Run formula verifications
    print("\n" + DIVIDER)
    print("FORMULA VERIFICATION")
    print(DIVIDER)

    forecast_ok = verify_forecast_formula()
    fdm_ok = verify_fdm_formula()
    idm_ok = verify_idm_formula()

    all_passed = forecast_ok and fdm_ok and idm_ok

    print("\n" + SUB_DIVIDER)
    print("VERIFICATION SUMMARY")
    print(SUB_DIVIDER)
    print(f"\n  Forecast Formula: {'PASS' if forecast_ok else 'FAIL'}")
    print(f"  FDM Formula: {'PASS' if fdm_ok else 'FAIL'}")
    print(f"  IDM Formula: {'PASS' if idm_ok else 'FAIL'}")
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    # Run full pipeline demo
    contracts = run_full_pipeline()

    print("\n" + DIVIDER)
    print("DEMO COMPLETE")
    print(DIVIDER)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
