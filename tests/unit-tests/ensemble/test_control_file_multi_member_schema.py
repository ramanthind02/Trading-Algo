"""
Tests for multi-member schema validation in control files.

This module tests that:
1. The 'members' array is required in base model configurations
2. Legacy schema (without members) is rejected
"""

import pytest
import json
import tempfile
from pathlib import Path

from ensemble.ensemble_utils import (
    validate_control_file,
    parse_control_file,
    validate_base_model_config,
)


class TestControlFileMultiMemberSchema:
    """Test suite for multi-member schema enforcement in control files."""

    def test_base_model_config_requires_members_array(self):
        """Test that base model config must have a 'members' array."""
        config = {
            'name': 'test_model',
            'model_type': 'continuous_binning',
            'feature_column': 'test_feature',
            'strategy': 'long',
            'constructor_params': {'n_bins': 3},
            # Missing 'members' key - should fail
        }
        
        with pytest.raises(ValueError, match="members"):
            validate_base_model_config(config, index=0)

    def test_base_model_members_must_be_non_empty_list(self):
        """Test that 'members' must be a non-empty list."""
        config = {
            'name': 'test_model',
            'model_type': 'continuous_binning',
            'feature_column': 'test_feature',
            'strategy': 'long',
            'constructor_params': {'n_bins': 3},
            'members': [],  # Empty list - should fail
        }
        
        with pytest.raises(ValueError, match="members"):
            validate_base_model_config(config, index=0)

    def test_base_model_members_must_be_list(self):
        """Test that 'members' must be a list."""
        config = {
            'name': 'test_model',
            'model_type': 'continuous_binning',
            'feature_column': 'test_feature',
            'strategy': 'long',
            'constructor_params': {'n_bins': 3},
            'members': 'not_a_list',  # String instead of list - should fail
        }
        
        with pytest.raises(ValueError, match="members"):
            validate_base_model_config(config, index=0)

    def test_valid_base_model_config_with_members(self):
        """Test that valid config with members passes validation."""
        config = {
            'name': 'test_model',
            'model_type': 'continuous_binning',
            'feature_column': 'test_feature',
            'strategy': 'long',
            'constructor_params': {'n_bins': 3},
            'members': [
                {'member_name': 'member_1', 'params': {'bin_index': 0}},
                {'member_name': 'member_2', 'params': {'bin_index': 1}},
            ]
        }
        
        # Should not raise
        validate_base_model_config(config, index=0)

    def test_control_file_rejects_legacy_schema_without_members(self):
        """Test that control file rejects legacy schema without 'members' in base models."""
        control_file = {
            'metadata': {
                'is_fit': False,
                'version': '2.0.0'
            },
            'base_models': [
                {
                    'name': 'legacy_model',
                    'model_type': 'continuous_binning',
                    'feature_column': 'test_feature',
                    'strategy': 'long',
                    'constructor_params': {'n_bins': 3}
                    # No 'members' - legacy schema
                }
            ]
        }
        
        with pytest.raises(ValueError, match="members"):
            validate_control_file(control_file)

    def test_control_file_accepts_multi_member_schema(self):
        """Test that control file accepts multi-member schema."""
        control_file = {
            'metadata': {
                'is_fit': False,
                'version': '2.0.0'
            },
            'base_models': [
                {
                    'name': 'test_model',
                    'model_type': 'continuous_binning',
                    'feature_column': 'test_feature',
                    'strategy': 'long',
                    'constructor_params': {'n_bins': 3},
                    'members': [
                        {'member_name': 'member_1', 'params': {'bin_index': 0}},
                        {'member_name': 'member_2', 'params': {'bin_index': 1}},
                    ]
                }
            ]
        }
        
        # Should not raise
        validate_control_file(control_file)

    def test_parse_control_file_with_multi_member_schema(self):
        """Test parsing control file with multi-member schema."""
        control_file = {
            'metadata': {
                'is_fit': False,
                'version': '2.0.0'
            },
            'base_models': [
                {
                    'name': 'test_model',
                    'model_type': 'continuous_binning',
                    'feature_column': 'test_feature',
                    'strategy': 'long',
                    'constructor_params': {'n_bins': 3},
                    'members': [
                        {'member_name': 'member_1', 'params': {'bin_index': 0}},
                    ]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(control_file, f)
            filepath = f.name
        
        try:
            result = parse_control_file(filepath)
            assert 'base_models' in result
            assert len(result['base_models']) == 1
            assert 'members' in result['base_models'][0]
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_control_file_with_is_fit_true_validates_members_in_fitted(self):
        """Test that control file with is_fit=True validates fitted_ensemble has members info."""
        control_file = {
            'metadata': {
                'is_fit': True,
                'version': '2.0.0',
                'selection_method': 'manual'
            },
            'base_models': [
                {
                    'name': 'test_model',
                    'model_type': 'continuous_binning',
                    'feature_column': 'test_feature',
                    'strategy': 'long',
                    'constructor_params': {'n_bins': 3},
                    'members': [
                        {'member_name': 'member_1', 'params': {'bin_index': 0}},
                    ]
                }
            ],
            'fitted_base_models': {
                'test_model': {
                    'member_1': {'thresholds': [0.5]}
                }
            },
            'fitted_ensemble': {
                'weights': {'test_model::member_1': 1.0},
                'exposure_fractions': {'test_model::member_1': 0.5},
                'feature_names': ['test_model::member_1'],
                'target_volatility': 0.15,
                'unique_tickers': ['ES'],
                'instrument_weights': {'ES': 1.0},
                'n_tickers': 1
            }
        }
        
        # Should not raise - this should work with multi-member
        validate_control_file(control_file)
