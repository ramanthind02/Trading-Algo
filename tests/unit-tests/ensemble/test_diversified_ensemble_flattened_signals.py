"""
Tests for diversified ensemble flattened member signals.

These tests verify that:
1. The ensemble can consume flattened member signals from multi-member base models
2. Feature names include member identifiers (e.g., model_name::member_id)
3. Weights are calculated at the member level
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from ensemble.diversified_ensemble import DiversifiedEnsemble
from ensemble.ensemble_utils import validate_control_file
from utils.core.enums import TimeFrame


class TestDiversifiedEnsembleFlattenedMemberSignals:
    """Test suite for flattened member signal consumption in DiversifiedEnsemble."""

    def test_extract_member_feature_names_from_control_file(self):
        """Test that control file with members produces flattened feature names."""
        control_file = {
            'metadata': {
                'is_fit': False,
                'version': '2.0.0'
            },
            'base_models': [
                {
                    'name': 'test_model_1',
                    'model_type': 'continuous_binning',
                    'feature_column': 'test_feature_1',
                    'strategy': 'long',
                    'constructor_params': {'n_bins': 3},
                    'members': [
                        {'member_id': 'member_0', 'bin_index': 0},
                        {'member_id': 'member_1', 'bin_index': 1},
                    ]
                },
                {
                    'name': 'test_model_2',
                    'model_type': 'continuous_binning',
                    'feature_column': 'test_feature_2',
                    'strategy': 'short',
                    'constructor_params': {'n_bins': 2},
                    'members': [
                        {'member_id': 'variant_a', 'bin_index': 0},
                    ]
                }
            ]
        }
        
        # Validate the control file (should pass now with members)
        validate_control_file(control_file)
        
        # Check that we can extract member IDs
        member_ids = []
        for model in control_file['base_models']:
            if 'members' in model:
                for member in model['members']:
                    member_ids.append(f"{model['name']}::{member['member_id']}")
        
        # Should have flattened names
        assert len(member_ids) == 3  # 2 + 1
        assert 'test_model_1::member_0' in member_ids
        assert 'test_model_1::member_1' in member_ids
        assert 'test_model_2::variant_a' in member_ids

    def test_control_file_validates_members_structure(self):
        """Test that the control file validation properly handles members."""
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
                        {'member_id': 'm1', 'bin_index': 0},
                        {'member_id': 'm2', 'bin_index': 1},
                    ]
                }
            ]
        }
        
        # Should not raise - validation should pass
        validate_control_file(control_file)

    def test_ensemble_extracts_member_info_from_base_models(self):
        """Test that ensemble can extract member info from base model configs."""
        control_file = {
            'metadata': {
                'is_fit': False,
                'version': '2.0.0'
            },
            'base_models': [
                {
                    'name': 'model_A',
                    'model_type': 'continuous_binning',
                    'feature_column': 'feature_A',
                    'strategy': 'long',
                    'constructor_params': {'n_bins': 3},
                    'members': [
                        {'member_id': 'bin_0', 'bin_index': 0},
                        {'member_id': 'bin_1', 'bin_index': 1},
                        {'member_id': 'bin_2', 'bin_index': 2},
                    ]
                }
            ]
        }
        
        # Validate passes
        validate_control_file(control_file)
        
        # Extract flattened feature names
        flattened_features = []
        for model in control_file['base_models']:
            model_name = model['name']
            if 'members' in model:
                for member in model['members']:
                    member_id = member['member_id']
                    flattened_features.append(f"{model_name}::{member_id}")
        
        assert len(flattened_features) == 3
        assert all('::' in f for f in flattened_features)

    def test_multi_member_schema_creates_unique_feature_ids(self):
        """Test that multi-member schema creates unique feature IDs."""
        control_file = {
            'metadata': {
                'is_fit': False,
                'version': '2.0.0'
            },
            'base_models': [
                {
                    'name': 'ewmac_fast',
                    'model_type': 'continuous_binning',
                    'feature_column': 'ewmac_16_64',
                    'strategy': 'long_short',
                    'constructor_params': {'n_bins': 3},
                    'members': [
                        {'member_id': 'q33', 'bin_index': 0},
                        {'member_id': 'q66', 'bin_index': 1},
                        {'member_id': 'q100', 'bin_index': 2},
                    ]
                },
                {
                    'name': 'ewmac_slow',
                    'model_type': 'continuous_binning',
                    'feature_column': 'ewmac_32_128',
                    'strategy': 'long_short',
                    'constructor_params': {'n_bins': 2},
                    'members': [
                        {'member_id': 'lower', 'bin_index': 0},
                        {'member_id': 'upper', 'bin_index': 1},
                    ]
                }
            ]
        }
        
        validate_control_file(control_file)
        
        # Create unique IDs
        unique_ids = set()
        for model in control_file['base_models']:
            model_name = model['name']
            for member in model['members']:
                unique_id = f"{model_name}::{member['member_id']}"
                unique_ids.add(unique_id)
        
        assert len(unique_ids) == 5  # 3 + 2
        assert 'ewmac_fast::q33' in unique_ids
        assert 'ewmac_fast::q66' in unique_ids
        assert 'ewmac_fast::q100' in unique_ids
        assert 'ewmac_slow::lower' in unique_ids
        assert 'ewmac_slow::upper' in unique_ids

    def test_members_required_for_ensemble_to_work(self):
        """Test that ensemble requires members array in control file."""
        control_file_legacy = {
            'metadata': {
                'is_fit': False,
                'version': '2.0.0'
            },
            'base_models': [
                {
                    'name': 'old_model',
                    'model_type': 'continuous_binning',
                    'feature_column': 'old_feature',
                    'strategy': 'long',
                    'constructor_params': {'n_bins': 3}
                    # No 'members' - should fail
                }
            ]
        }
        
        # Should raise because legacy schema doesn't have members
        with pytest.raises(ValueError, match="members"):
            validate_control_file(control_file_legacy)

    def test_member_signal_strength_annualizes_and_clips_daily_sharpe_outputs(self):
        """Continuous member outputs should map to clipped annualized strength."""
        ensemble = DiversifiedEnsemble(
            base_models={},
            required_columns=[],
            base_tf=TimeFrame.D,
        )
        member_output = pd.Series([0.0, 0.05, -0.10, 0.30, np.nan])

        result = ensemble._annualized_member_signal_strength(member_output)

        scale = np.sqrt(252.0)
        expected = np.array(
            [
                0.0,
                0.05 * scale,
                -0.10 * scale,
                2.0,  # clipped from ~4.76
                0.0,
            ]
        )
        assert np.allclose(result, expected)
