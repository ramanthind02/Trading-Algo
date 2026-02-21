"""
Tests for multi-member BaseModel functionality.

These tests verify that BaseModel can own multiple binning members and emit
flattened member-level signal outputs using build_member_model_name.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta

from feature_selection.base_models.feature_base_model import BaseModel, build_member_model_name
from feature_selection.base_models.continuous_binning import ContinuousBinningModel as QuantileBinningModel
from utils.enums import Ticker, TimeFrame


class TestBuildMemberModelName:
    """Test the build_member_model_name helper function."""

    def test_with_member_identity(self):
        """Test that member identity is appended with :: separator."""
        result = build_member_model_name("base_model", "member_1")
        assert result == "base_model::member_1"

    def test_with_empty_member_identity(self):
        """Test that empty member identity returns base model name."""
        result = build_member_model_name("base_model", "")
        assert result == "base_model"

    def test_with_none_member_identity(self):
        """Test that None member identity returns base model name."""
        result = build_member_model_name("base_model", None)
        assert result == "base_model"


class TestMultiMemberBaseModel:
    """Test that BaseModel can hold multiple members and emit flattened signals."""

    def test_base_model_has_members_attribute(self):
        """Test that BaseModel has a members container attribute."""
        with patch('feature_selection.base_models.feature_base_model.helpers.create_bias_node') as mock_create:
            mock_node = MagicMock()
            mock_node.output_features = ['signal']
            mock_create.return_value = mock_node
            
            feature_config = {
                'bias_node_spec': {
                    'module_name': 'ewmac',
                    'timeframes': [TimeFrame.D],
                    'params': {'fastSpan': 16, 'slowSpan': 64}
                },
                'model_type': 'QuantileBinningModel',
                'constructor_params': {'n_bins': 3}
            }
            
            model = BaseModel(
                feature_config=feature_config,
                tickers=[Ticker.ES],
                use_cache=False
            )
            
            # BaseModel should have a members attribute for storing multiple binning models
            assert hasattr(model, 'members'), "BaseModel should have 'members' attribute"

    def test_base_model_members_initially_empty(self):
        """Test that members container starts empty."""
        with patch('feature_selection.base_models.feature_base_model.helpers.create_bias_node') as mock_create:
            mock_node = MagicMock()
            mock_node.output_features = ['signal']
            mock_create.return_value = mock_node
            
            feature_config = {
                'bias_node_spec': {
                    'module_name': 'ewmac',
                    'timeframes': [TimeFrame.D],
                    'params': {'fastSpan': 16, 'slowSpan': 64}
                },
                'model_type': 'QuantileBinningModel',
                'constructor_params': {'n_bins': 3}
            }
            
            model = BaseModel(
                feature_config=feature_config,
                tickers=[Ticker.ES],
                use_cache=False
            )
            
            # Members should be empty initially
            assert model.members == [], "members should be empty list"

    def test_add_member_to_base_model(self):
        """Test that we can add a member (binning model) to BaseModel."""
        with patch('feature_selection.base_models.feature_base_model.helpers.create_bias_node') as mock_create:
            mock_node = MagicMock()
            mock_node.output_features = ['signal']
            mock_create.return_value = mock_node
            
            feature_config = {
                'bias_node_spec': {
                    'module_name': 'ewmac',
                    'timeframes': [TimeFrame.D],
                    'params': {'fastSpan': 16, 'slowSpan': 64}
                },
                'model_type': 'QuantileBinningModel',
                'constructor_params': {'n_bins': 3}
            }
            
            model = BaseModel(
                feature_config=feature_config,
                tickers=[Ticker.ES],
                use_cache=False
            )
            
            # Create and add a binning member
            binning_member = QuantileBinningModel(n_bins=3)
            member_name = build_member_model_name("ewmac_1H_signal", "variant_1")
            
            # Add member using public API
            model.add_member(member_name, binning_member)
            
            assert len(model.members) == 1
            assert model.members[0][0] == member_name

    def test_fit_multiple_members(self):
        """Test that BaseModel can fit multiple members on the same data."""
        with patch('feature_selection.base_models.feature_base_model.helpers.create_bias_node') as mock_create:
            mock_node = MagicMock()
            mock_node.output_features = ['signal']
            mock_create.return_value = mock_node
            
            feature_config = {
                'bias_node_spec': {
                    'module_name': 'ewmac',
                    'timeframes': [TimeFrame.D],
                    'params': {'fastSpan': 16, 'slowSpan': 64}
                },
                'model_type': 'QuantileBinningModel',
                'constructor_params': {'n_bins': 3}
            }
            
            model = BaseModel(
                feature_config=feature_config,
                tickers=[Ticker.ES],
                use_cache=False
            )
            
            # Add multiple members using public API
            for i in range(3):
                binning_member = QuantileBinningModel(n_bins=3)
                member_name = build_member_model_name("ewmac_1H_signal", f"variant_{i}")
                model.add_member(member_name, binning_member)
            
            assert len(model.members) == 3

    def test_emit_flattened_member_signals(self):
        """Test that BaseModel emits flattened member-level signals after prediction."""
        with patch('feature_selection.base_models.feature_base_model.helpers.create_bias_node') as mock_create:
            mock_node = MagicMock()
            mock_node.output_features = ['signal']
            mock_create.return_value = mock_node
            
            feature_config = {
                'bias_node_spec': {
                    'module_name': 'ewmac',
                    'timeframes': [TimeFrame.D],
                    'params': {'fastSpan': 16, 'slowSpan': 64}
                },
                'model_type': 'QuantileBinningModel',
                'constructor_params': {'n_bins': 3}
            }
            
            model = BaseModel(
                feature_config=feature_config,
                tickers=[Ticker.ES],
                use_cache=False
            )
            
            # Add multiple members using public API
            for i in range(3):
                binning_member = QuantileBinningModel(n_bins=3)
                member_name = build_member_model_name("ewmac_1H_signal", f"variant_{i}")
                model.add_member(member_name, binning_member)
            
            # BaseModel should have a method to emit flattened member signals
            assert hasattr(model, 'emit_member_signals'), \
                "BaseModel should have 'emit_member_signals' method"

    def test_member_signals_returns_dataframe(self):
        """Test that emit_member_signals returns a DataFrame with proper structure."""
        with patch('feature_selection.base_models.feature_base_model.helpers.create_bias_node') as mock_create:
            mock_node = MagicMock()
            mock_node.output_features = ['signal']
            mock_create.return_value = mock_node
            
            feature_config = {
                'bias_node_spec': {
                    'module_name': 'ewmac',
                    'timeframes': [TimeFrame.D],
                    'params': {'fastSpan': 16, 'slowSpan': 64}
                },
                'model_type': 'QuantileBinningModel',
                'constructor_params': {'n_bins': 3}
            }
            
            model = BaseModel(
                feature_config=feature_config,
                tickers=[Ticker.ES],
                use_cache=False
            )
            
            # Add multiple members and mock them as fitted
            dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
            feature_data = pd.Series(np.random.randn(50), index=dates, name="ewmac_signal")
            target_data = pd.Series(np.random.randn(50) * 0.01, index=dates)
            
            for i in range(3):
                binning_member = QuantileBinningModel(n_bins=3)
                binning_member.fit(feature_data, target_data)
                member_name = build_member_model_name("ewmac_signal", f"variant_{i}")
                model.add_member(member_name, binning_member)
            
            # emit_member_signals should return a DataFrame with columns for each member
            result = model.emit_member_signals(feature_data=feature_data, strategy='long')
            
            assert isinstance(result, pd.DataFrame), \
                "emit_member_signals should return a DataFrame"
            
            # Check that all member names are in the columns
            for member_name, _ in model.members:
                assert member_name in result.columns, \
                    f"Member {member_name} should be in output columns"

    def test_single_member_backward_compatible(self):
        """Test that single member mode works like original BaseModel."""
        with patch('feature_selection.base_models.feature_base_model.helpers.create_bias_node') as mock_create:
            mock_node = MagicMock()
            mock_node.output_features = ['signal']
            mock_create.return_value = mock_node
            
            feature_config = {
                'bias_node_spec': {
                    'module_name': 'ewmac',
                    'timeframes': [TimeFrame.D],
                    'params': {'fastSpan': 16, 'slowSpan': 64}
                },
                'model_type': 'QuantileBinningModel',
                'constructor_params': {'n_bins': 3}
            }
            
            model = BaseModel(
                feature_config=feature_config,
                tickers=[Ticker.ES],
                use_cache=False
            )
            
            # With no explicit members, should use binning_model for backward compat
            # Original behavior: model.binning_model should work
            assert hasattr(model, 'binning_model')
            assert model.binning_model is not None
