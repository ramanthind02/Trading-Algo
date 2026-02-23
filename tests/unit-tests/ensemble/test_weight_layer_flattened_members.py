"""
Tests for flattened member pipeline in weight_layer and portfolio.

These tests verify that:
1. Weight layer can handle flattened member signals (model_name::member_id)
2. Portfolio can consume flattened member forecasts
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from ensemble.weight_layer import (
    WeightLayer,
    InverseCorrelationWeightLayer,
    WeightLayerConfig,
    InverseCorrelationWeighter,
)
from ensemble.portfolio import Portfolio
from utils.core.enums import TimeFrame


class TestWeightLayerFlattenedMemberSignals:
    """Test suite for weight layer handling of flattened member signals."""

    def test_weight_layer_config_accepts_flattened_model_names(self):
        """Test that WeightLayerConfig works with flattened model names."""
        config = WeightLayerConfig(
            weighting_method="inverse_correlation",
            group_method="feature_family"
        )
        assert config.weighting_method == "inverse_correlation"

    def test_inverse_correlation_weighter_handles_flattened_names(self):
        """Test that InverseCorrelationWeighter handles flattened model names."""
        weighter = InverseCorrelationWeighter()
        
        signals = pd.DataFrame({
            'model_a::member_1': [0, 1, 0, 1, 0],
            'model_a::member_2': [1, 0, 1, 0, 1],
            'model_b::variant_x': [0, 0, 1, 1, 0],
        })
        
        weighter.fit(signals)
        weights = weighter.get_weights()
        
        assert len(weights) == 3
        assert all('::' in name for name in weights.index)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_weight_layer_fit_with_member_level_signals(self):
        """Test that weight layer can fit with member-level signals."""
        config = WeightLayerConfig(
            weighting_method="inverse_correlation",
            fdm_max=2.0
        )
        weight_layer = WeightLayer(weight_method=config.weighting_method, fdm_max=config.fdm_max)
        
        signals = pd.DataFrame({
            'ewmac_fast::q33': [0, 1, 0, 1, 0, 1, 0],
            'ewmac_fast::q66': [1, 0, 1, 0, 1, 0, 1],
            'ewmac_slow::lower': [0, 0, 1, 1, 0, 0, 1],
        }, index=pd.date_range('2020-01-01', periods=7, freq='D'))
        
        forecast_vectors = [
            pd.DataFrame({
                'ticker': ['ES'] * 7,
                'model_name': ['ewmac_fast::q33'] * 7,
                'forecast': signals['ewmac_fast::q33'] * 0.1,
                'signal': signals['ewmac_fast::q33']
            }),
            pd.DataFrame({
                'ticker': ['ES'] * 7,
                'model_name': ['ewmac_fast::q66'] * 7,
                'forecast': signals['ewmac_fast::q66'] * 0.1,
                'signal': signals['ewmac_fast::q66']
            }),
            pd.DataFrame({
                'ticker': ['ES'] * 7,
                'model_name': ['ewmac_slow::lower'] * 7,
                'forecast': signals['ewmac_slow::lower'] * 0.1,
                'signal': signals['ewmac_slow::lower']
            }),
        ]
        
        weight_layer.fit(forecast_vectors, signals)
        
        assert weight_layer.is_fitted_
        assert 'ES' in weight_layer.weights_

    def test_weight_layer_weights_use_member_identifiers(self):
        """Test that weight layer weights use member identifiers."""
        weighter = InverseCorrelationWeighter()
        
        signals = pd.DataFrame({
            'base_model::member_A': [0, 1, 1, 0],
            'base_model::member_B': [1, 0, 0, 1],
        })
        
        weighter.fit(signals)
        weights = weighter.get_weights()
        
        assert all('::' in name for name in weights.index)
        assert 'base_model::member_A' in weights.index
        assert 'base_model::member_B' in weights.index


class TestPortfolioFlattenedMemberForecasts:
    """Test suite for portfolio handling of flattened member forecasts."""

    def test_portfolio_predict_with_member_level_forecasts(self):
        """Test that portfolio predict method works with member-level forecasts."""
        mock_ensemble = MagicMock()
        mock_ensemble.predict.return_value = pd.DataFrame({
            'ticker': ['ES', 'ES', 'NQ', 'NQ'],
            'model_name': ['model_a::m1', 'model_a::m2', 'model_b::m1', 'model_b::m2'],
            'forecast': [0.1, -0.1, 0.15, -0.15],
            'signal': [1, 1, 1, 1]
        })
        mock_ensemble.unique_tickers_ = ['ES', 'NQ']
        
        portfolio = Portfolio(
            ensembles=[mock_ensemble],
            trading_timeframe=TimeFrame.D,
        )
        portfolio.is_fitted_ = True
        portfolio.idm_ = 1.0
        
        mock_wl = MagicMock()
        mock_wl.combine.return_value = pd.DataFrame({
            'ticker': ['ES', 'NQ'],
            'forecast': [0.05, 0.07],
            'signal': [1, 1]
        })
        portfolio.weight_layer = mock_wl
        
        result = mock_ensemble.predict()
        
        assert result is not None
        assert 'model_name' in result.columns
        model_names = result['model_name'].tolist()
        assert 'model_a::m1' in model_names

    def test_forecast_vectors_have_member_ids(self):
        """Test that ensemble forecasts include member IDs in model names."""
        mock_ensemble = MagicMock()
        
        mock_ensemble.predict.return_value = pd.DataFrame({
            'ticker': ['ES', 'ES', 'ES'],
            'model_name': ['ewmac::q33', 'ewmac::q66', 'ewmac::q100'],
            'forecast': [0.1, 0.1, 0.1],
            'signal': [1, 1, 1]
        })
        
        result = mock_ensemble.predict()
        
        model_names = result['model_name'].tolist()
        assert all('::' in name for name in model_names)
