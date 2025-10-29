"""
Demo: Using FeatureExtractor for Centralized Feature Extraction

This script demonstrates how to use the new FeatureExtractor class to extract
features from multiple bias nodes with parameter grids, then consume those
features in FeatureExplorer and FeatureSelector.

The FeatureExtractor follows the Single Responsibility Principle:
- It ONLY handles feature extraction
- It returns clean dataframes
- FeatureExplorer and FeatureSelector consume those dataframes
- No coupling between extraction and analysis

Author: Trading Research Team
Date: 2025-10-28
"""

from datetime import datetime as dt
from utils.enums import Ticker, TimeFrame
from feature_extraction.feature_extractor_class import FeatureExtractor


def demo_single_ticker_single_module():
    """Basic usage: Single ticker, single module with parameter grid."""
    print("\n" + "="*70)
    print("DEMO 1: Single Ticker, Single Module")
    print("="*70)
    
    # Create extractor
    extractor = FeatureExtractor(
        tickers=Ticker.SPY,
        start=dt(2010, 1, 1),
        end=dt(2020, 12, 31)
    )
    
    # Extract RSI with multiple lookback periods
    features_dict, targets_dict, combined_features, combined_targets = extractor.extract(
        modules={
            'rsi': {'lookback': [14, 21, 28]}
        }
    )
    
    # Show results
    print("\n" + "-"*70)
    print("Extraction Results:")
    print("-"*70)
    print(extractor.summary())
    
    # Access per-module features
    rsi_features, rsi_targets = extractor.get_module_features('rsi')
    print(f"\nRSI features shape: {rsi_features.shape}")
    print(f"RSI features columns: {list(rsi_features.columns)[:5]}...")
    
    # Access combined features
    print(f"\nCombined features shape: {combined_features.shape}")
    print(f"Combined targets columns: {list(combined_targets.columns)}")
    
    return extractor


def demo_single_ticker_multi_module():
    """Extract from multiple modules with parameter grids."""
    print("\n" + "="*70)
    print("DEMO 2: Single Ticker, Multiple Modules")
    print("="*70)
    
    # Create extractor
    extractor = FeatureExtractor(
        tickers=Ticker.SPY,
        start=dt(2015, 1, 1),
        end=dt(2020, 12, 31)
    )
    
    # Extract from multiple modules with parameter grids
    features_dict, targets_dict, combined_features, combined_targets = extractor.extract(
        modules={
            'rsi': {'lookback': [14, 21]},
            'cmma': {'lookback': [20, 50], 'atr_length': [252]},
            'volatility_regime': {'window': [20, 60], 'lookback': [252]}
        }
    )
    
    # Show results
    print("\n" + "-"*70)
    print("Extraction Results:")
    print("-"*70)
    print(extractor.summary())
    
    # Access per-module features
    for module_name in features_dict.keys():
        features_df, targets_df = extractor.get_module_features(module_name)
        n_features = len([col for col in features_df.columns if col != 'ticker'])
        print(f"\n{module_name}: {n_features} features")
        print(f"  Columns: {list(features_df.columns)[:3]}...")
    
    return extractor


def demo_multi_ticker_multi_module():
    """Extract from multiple tickers and modules."""
    print("\n" + "="*70)
    print("DEMO 3: Multiple Tickers, Multiple Modules")
    print("="*70)
    
    # Create extractor with multiple tickers
    extractor = FeatureExtractor(
        tickers=[Ticker.ES, Ticker.NQ, Ticker.YM],
        start=dt(2015, 1, 1),
        end=dt(2020, 12, 31)
    )
    
    # Extract from multiple modules
    features_dict, targets_dict, combined_features, combined_targets = extractor.extract(
        modules={
            'rsi': {'lookback': [14]},
            'cmma': {'lookback': [20], 'atr_length': [252]}
        }
    )
    
    # Show results
    print("\n" + "-"*70)
    print("Extraction Results:")
    print("-"*70)
    print(extractor.summary())
    
    # Check ticker column
    if 'ticker' in combined_features.columns:
        print(f"\nTickers in data: {combined_features['ticker'].unique()}")
        print(f"Samples per ticker:")
        print(combined_features.groupby('ticker').size())
    
    return extractor


def demo_integration_with_explorer():
    """Show how to use extracted features with FeatureExplorer."""
    print("\n" + "="*70)
    print("DEMO 4: Integration with FeatureExplorer")
    print("="*70)
    
    # Step 1: Extract features using FeatureExtractor
    print("\nStep 1: Extract features...")
    extractor = FeatureExtractor(
        tickers=Ticker.SPY,
        start=dt(2015, 1, 1),
        end=dt(2020, 12, 31)
    )
    
    features_dict, targets_dict, combined_features, combined_targets = extractor.extract(
        modules={
            'rsi': {'lookback': [14]}
        }
    )
    
    # Step 2: Create FeatureExplorer from extracted dataframes
    print("\nStep 2: Create FeatureExplorer from dataframes...")
    from feature_selection.feature_explorer import FeatureExplorer
    
    # Get RSI features
    rsi_features, rsi_targets = extractor.get_module_features('rsi')
    
    # Create explorer for first feature
    feature_name = [col for col in rsi_features.columns if col != 'ticker'][0]
    feature_data = rsi_features[feature_name]
    
    explorer = FeatureExplorer(
        feature_name=feature_name,
        feature_data=feature_data,
        target_data=rsi_targets['log_return'],  # Use log_return as target
        raw_return=rsi_targets['raw_return'],
        log_return=rsi_targets['log_return'],
        log_return_atr=rsi_targets['log_return_atr'],
        log_return_ewsd=rsi_targets['log_return_ewsd']
    )
    
    print(f"\n✓ Created FeatureExplorer: {explorer}")
    print(f"  Feature: {feature_name}")
    print(f"  Samples: {explorer.n_samples}")
    print(f"  Date range: {explorer.date_range}")
    
    # Step 3: Use FeatureExplorer methods
    print("\nStep 3: Use FeatureExplorer methods...")
    print("  (Explorer is now decoupled from extraction logic!)")
    
    # Example: plot deciles
    # fig = explorer.plot_feature_deciles(n_bins=10)
    
    return extractor, explorer


def demo_integration_with_selector():
    """Show how to use extracted features with FeatureSelector."""
    print("\n" + "="*70)
    print("DEMO 5: Integration with FeatureSelector")
    print("="*70)
    
    # Step 1: Extract features using FeatureExtractor
    print("\nStep 1: Extract features...")
    extractor = FeatureExtractor(
        tickers=Ticker.SPY,
        start=dt(2010, 1, 1),
        end=dt(2020, 12, 31)
    )
    
    features_dict, targets_dict, combined_features, combined_targets = extractor.extract(
        modules={
            'cmma': {'lookback': [20], 'atr_length': [252]}
        }
    )
    
    # Step 2: Create FeatureSelector from extracted dataframes
    print("\nStep 2: Create FeatureSelector from dataframes...")
    from feature_selection.feature_selector import FeatureSelector
    
    # Get CMMA features
    cmma_features, cmma_targets = extractor.get_module_features('cmma')
    
    # Create selector for first feature
    feature_name = [col for col in cmma_features.columns if col != 'ticker'][0]
    feature_data = cmma_features[feature_name]
    
    selector = FeatureSelector(
        feature_name=feature_name,
        feature_data=feature_data,
        target_data=cmma_targets  # Pass all target columns
    )
    
    print(f"\n✓ Created FeatureSelector: {selector}")
    print(f"  Feature: {feature_name}")
    print(f"  Samples: {selector.n_samples}")
    print(f"  Date range: {selector.date_range}")
    
    # Step 3: Use FeatureSelector methods
    print("\nStep 3: Use FeatureSelector methods...")
    print("  (Selector is now decoupled from extraction logic!)")
    
    # Example: walkforward analysis
    # from feature_selection.base_models import QuantileBinningModel
    # from feature_selection.objective_metric import SortinoRatio
    # model = QuantileBinningModel(n_bins=3)
    # metric = SortinoRatio()
    # results = selector.walkforward_analysis(model, metric, ...)
    
    return extractor, selector


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("FeatureExtractor Demonstration")
    print("Centralized Feature Extraction with Single Responsibility")
    print("="*70)
    
    # Run demos
    demo_single_ticker_single_module()
    demo_single_ticker_multi_module()
    demo_multi_ticker_multi_module()
    demo_integration_with_explorer()
    demo_integration_with_selector()
    
    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. FeatureExtractor handles ALL extraction logic")
    print("2. Returns clean dataframes (per-module + combined)")
    print("3. FeatureExplorer and FeatureSelector consume dataframes")
    print("4. No coupling between extraction and analysis")
    print("5. Easy to explore parameter grids")
    print("="*70)


if __name__ == '__main__':
    main()
