"""
Tests for OSFeatureSelector

Basic unit tests to verify the OSFeatureSelector implementation works correctly.

Author: Trading Research Team
Date: 2025-11-28
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from feature_selection.os_feature_selector import OSFeatureSelector


class TestOSFeatureSelector:
    """Test suite for OSFeatureSelector."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample features and targets data for testing."""
        # Create 500 days of sample data
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # Create sample features
        np.random.seed(42)
        features_df = pd.DataFrame({
            'rsi_signal_D_lookback_14': np.random.randn(500),
            'momentum_signal_D_lookback_20': np.random.randn(500),
            'ma_cross_signal_D_lookback_10': np.random.choice([0, 1], 500)
        }, index=dates)
        
        # Create sample targets
        targets_df = pd.DataFrame({
            'raw_return': np.random.randn(500) * 0.02,
            'log_return': np.random.randn(500) * 0.02,
            'log_return_atr': np.random.randn(500) * 0.02,
            'log_return_ewsd': np.random.randn(500) * 0.02
        }, index=dates)
        
        return features_df, targets_df
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock BaseModel for testing."""
        model = Mock()
        model.__class__.__name__ = 'MockModel'
        
        # Mock fit method
        model.fit = Mock()
        
        # Mock predict method - return random binary signals
        def mock_predict(X, strategy='long', normalization_data=None):
            return pd.Series(np.random.choice([0, 1], len(X)), index=X.index)
        model.predict = Mock(side_effect=mock_predict)
        
        return model
    
    @pytest.fixture
    def mock_metric(self):
        """Create a mock ObjectiveMetric for testing."""
        metric = Mock()
        metric.__class__.__name__ = 'MockMetric'
        metric.compute = Mock(return_value=1.5)  # Return fixed value for testing
        return metric
    
    def test_initialization(self, sample_data):
        """Test OSFeatureSelector initialization."""
        features_df, targets_df = sample_data
        
        selector = OSFeatureSelector(features_df, targets_df)
        
        assert selector.n_features == 3
        assert selector.n_samples == 500
        assert len(selector.feature_names) == 3
        assert 'rsi_signal_D_lookback_14' in selector.feature_names
        assert 'momentum_signal_D_lookback_20' in selector.feature_names
        assert 'ma_cross_signal_D_lookback_10' in selector.feature_names
        assert not selector.has_ticker
        assert selector.results == {}
    
    def test_initialization_with_mismatched_indices(self, sample_data):
        """Test that initialization fails with mismatched indices."""
        features_df, targets_df = sample_data
        
        # Create targets with different dates
        different_dates = pd.date_range('2021-01-01', periods=500, freq='D')
        targets_df_different = targets_df.copy()
        targets_df_different.index = different_dates
        
        with pytest.raises(ValueError, match="features_df and targets_df must have matching indices"):
            OSFeatureSelector(features_df, targets_df_different)
    
    def test_walkforward_test_single_feature(self, sample_data, mock_model, mock_metric):
        """Test walk-forward test with single feature."""
        features_df, targets_df = sample_data
        selector = OSFeatureSelector(features_df, targets_df)
        
        # Test single feature
        results_df, step_info, fig = selector.walkforward_test(
            model=mock_model,
            objective_metric=mock_metric,
            feature_cols='rsi_signal_D_lookback_14',
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 6, 1),
            test_step=50,
            num_steps=3,
            verbose=False
        )
        
        # Verify results structure
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(step_info, list)
        assert len(step_info) > 0
        assert 'step' in results_df.columns
        assert 'test_metric' in results_df.columns
        assert 'n_trades' in results_df.columns
        
        # Verify model was called
        assert mock_model.fit.called
        assert mock_model.predict.called
        assert mock_metric.compute.called
        
        # Verify results stored
        assert 'walkforward_test' in selector.results
        assert 'summary_df' in selector.results['walkforward_test']
    
    def test_walkforward_test_multiple_features(self, sample_data, mock_model, mock_metric):
        """Test walk-forward test with multiple features."""
        features_df, targets_df = sample_data
        selector = OSFeatureSelector(features_df, targets_df)
        
        # Test multiple features
        results = selector.walkforward_test(
            model=mock_model,
            objective_metric=mock_metric,
            feature_cols=['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 6, 1),
            test_step=50,
            num_steps=2,
            verbose=False
        )
        
        # Verify results structure for multiple features
        assert isinstance(results, dict)
        assert 'rsi_signal_D_lookback_14' in results
        assert 'momentum_signal_D_lookback_20' in results
        
        # Each feature should have (results_df, step_info, fig) tuple
        for feature_name in results:
            results_df, step_info, fig = results[feature_name]
            assert isinstance(results_df, pd.DataFrame)
            assert isinstance(step_info, list)
        
        # Verify summary DataFrame
        summary_df = selector.results['walkforward_test']['summary_df']
        assert len(summary_df) == 2
        assert 'feature' in summary_df.columns
        assert 'final_oos_metric' in summary_df.columns
    
    def test_cv_test_single_feature(self, sample_data, mock_model, mock_metric):
        """Test cross-validation test with single feature."""
        features_df, targets_df = sample_data
        selector = OSFeatureSelector(features_df, targets_df)
        
        results_df, fold_info, fig = selector.cv_test(
            model=mock_model,
            objective_metric=mock_metric,
            feature_cols='rsi_signal_D_lookback_14',
            n_splits=3,
            verbose=False
        )
        
        # Verify results structure
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(fold_info, list)
        assert len(fold_info) == 3  # Should have 3 folds
        assert 'fold' in results_df.columns
        assert 'test_metric' in results_df.columns
        
        # Verify results stored
        assert 'cv_test' in selector.results
    
    def test_cv_test_multiple_features(self, sample_data, mock_model, mock_metric):
        """Test cross-validation test with multiple features."""
        features_df, targets_df = sample_data
        selector = OSFeatureSelector(features_df, targets_df)
        
        results = selector.cv_test(
            model=mock_model,
            objective_metric=mock_metric,
            feature_cols=['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
            n_splits=3,
            verbose=False
        )
        
        # Verify results structure for multiple features
        assert isinstance(results, dict)
        assert len(results) == 2
        
        # Verify summary DataFrame
        summary_df = selector.results['cv_test']['summary_df']
        assert len(summary_df) == 2
    
    def test_permutation_test(self, sample_data):
        """Test permutation test with mock objective function."""
        features_df, targets_df = sample_data
        selector = OSFeatureSelector(features_df, targets_df)
        
        # Create simple objective function
        def mock_objective_func(feature_data, target_data, normalization_data=None):
            return np.random.random()  # Return random value
        
        # Disable multiprocessing for this test since local functions can't be pickled
        results = selector.permutation_test(
            feature_cols=['rsi_signal_D_lookback_14'],
            objective_func=mock_objective_func,
            nreps=10,  # Small number for testing
            n_jobs=1,  # Use single process to avoid pickling issues with local function
            verbose=False
        )
        
        # Verify results structure
        assert isinstance(results, pd.DataFrame)
        assert 'feature' in results.columns
        assert 'original_criterion' in results.columns
        assert 'pval' in results.columns
        assert 'significant' in results.columns
        assert len(results) == 1
        
        # Verify results stored
        assert 'permutation_test' in selector.results
    
    def test_get_results(self, sample_data, mock_model, mock_metric):
        """Test get_results method."""
        features_df, targets_df = sample_data
        selector = OSFeatureSelector(features_df, targets_df)
        
        # Run a test first
        selector.walkforward_test(
            model=mock_model,
            objective_metric=mock_metric,
            feature_cols='rsi_signal_D_lookback_14',
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 6, 1),
            test_step=50,
            num_steps=2,
            verbose=False
        )
        
        # Test get_results
        results = selector.get_results('walkforward_test')
        assert isinstance(results, dict)
        assert 'summary_df' in results
        assert 'model' in results
        
        # Test with non-existent test
        with pytest.raises(ValueError, match="Test 'nonexistent' not found"):
            selector.get_results('nonexistent')
    
    def test_get_summary(self, sample_data, mock_model, mock_metric):
        """Test get_summary method."""
        features_df, targets_df = sample_data
        selector = OSFeatureSelector(features_df, targets_df)
        
        # Run multiple tests
        selector.walkforward_test(
            model=mock_model,
            objective_metric=mock_metric,
            feature_cols='rsi_signal_D_lookback_14',
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 6, 1),
            test_step=50,
            num_steps=2,
            verbose=False
        )
        
        selector.cv_test(
            model=mock_model,
            objective_metric=mock_metric,
            feature_cols='momentum_signal_D_lookback_20',
            n_splits=3,
            verbose=False
        )
        
        # Test summary
        summary = selector.get_summary()
        assert isinstance(summary, pd.DataFrame)
        assert 'test_name' in summary.columns
        assert 'feature' in summary.columns
        assert 'metric_value' in summary.columns
        assert len(summary) == 2  # One row per test
    
    def test_validation_methods(self, sample_data):
        """Test validation methods."""
        features_df, targets_df = sample_data
        selector = OSFeatureSelector(features_df, targets_df)
        
        # Test with invalid model (use a real object that doesn't have fit/predict)
        class InvalidModel:
            pass
        
        invalid_model = InvalidModel()
        invalid_metric = Mock()
        invalid_metric.compute = Mock(return_value=1.0)
        
        with pytest.raises(ValueError, match="Model must implement BaseModel interface"):
            selector._validate_model_and_metric(invalid_model, invalid_metric)
        
        # Test with invalid metric (use a real object that doesn't have compute)
        class InvalidMetric:
            pass
        
        valid_model = Mock()
        valid_model.fit = Mock()
        valid_model.predict = Mock()
        invalid_metric = InvalidMetric()  # Missing compute method
        
        with pytest.raises(ValueError, match="Objective metric must implement ObjectiveMetric interface"):
            selector._validate_model_and_metric(valid_model, invalid_metric)
    
    def test_string_representations(self, sample_data):
        """Test __str__ and __repr__ methods."""
        features_df, targets_df = sample_data
        selector = OSFeatureSelector(features_df, targets_df)
        
        # Test __repr__
        repr_str = repr(selector)
        assert 'OSFeatureSelector' in repr_str
        assert 'n_features=3' in repr_str
        assert 'n_samples=500' in repr_str
        
        # Test __str__
        str_repr = str(selector)
        assert 'OSFeatureSelector with 3 features' in str_repr
        assert 'rsi_signal_D_lookback_14' in str_repr


if __name__ == '__main__':
    pytest.main([__file__])
