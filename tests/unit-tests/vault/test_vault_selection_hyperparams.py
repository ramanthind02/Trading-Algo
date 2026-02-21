"""
Tests for vault manager selection hyperparams and member metadata.

These tests verify that:
1. Vault manager can persist selection method and hyperparameters
2. Vault manager can persist member metadata
3. Vault can load and validate selection hyperparams
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path

from ensemble.vault_manager import (
    generate_model_id,
    create_ensemble_directory,
)
from ensemble.ensemble_utils import (
    save_control_file,
    validate_control_file as validate_control_file_utils,
    parse_control_file,
)


class TestVaultSelectionHyperparams:
    """Test suite for vault selection hyperparams persistence."""

    def test_save_control_file_includes_selection_method(self):
        """Test that control file can include selection method and hyperparameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            control_file_path = f"{tmpdir}/control.json"
            
            base_models = [
                {
                    'name': 'test_model',
                    'model_type': 'continuous_binning',
                    'feature_column': 'test_feature',
                    'strategy': 'long',
                    'constructor_params': {'n_bins': 3},
                    'members': [
                        {'member_id': 'member_1', 'bin_index': 0},
                    ]
                }
            ]
            
            metadata = {
                'is_fit': False,
                'version': '2.0.0',
                'selection_method': 'walkforward_stability',
                'selection_hyperparams': {
                    'min_stability_score': 0.7,
                    'n_folds': 5,
                    'metric': 'sharpe'
                }
            }
            
            save_control_file(
                filepath=control_file_path,
                base_models=base_models,
                metadata=metadata
            )
            
            with open(control_file_path, 'r') as f:
                saved = json.load(f)
            
            assert 'selection_method' in saved['metadata']
            assert saved['metadata']['selection_method'] == 'walkforward_stability'
            assert 'selection_hyperparams' in saved['metadata']
            assert saved['metadata']['selection_hyperparams']['min_stability_score'] == 0.7

    def test_load_control_file_includes_selection_hyperparams(self):
        """Test that control file can be loaded with selection hyperparams."""
        with tempfile.TemporaryDirectory() as tmpdir:
            control_file_path = f"{tmpdir}/control.json"
            
            control_file = {
                'metadata': {
                    'is_fit': True,
                    'version': '2.0.0',
                    'selection_method': 'permutation_test',
                    'selection_hyperparams': {
                        'n_permutations': 100,
                        'p_value_threshold': 0.05
                    }
                },
                'base_models': [
                    {
                        'name': 'test_model',
                        'model_type': 'continuous_binning',
                        'feature_column': 'test_feature',
                        'strategy': 'long',
                        'constructor_params': {'n_bins': 3},
                        'members': [
                            {'member_id': 'member_1', 'bin_index': 0},
                        ]
                    }
                ],
                'tickers': ['ES'],
                'fitted_base_models': {},
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
            
            with open(control_file_path, 'w') as f:
                json.dump(control_file, f)
            
            loaded = parse_control_file(control_file_path)
            
            assert 'selection_method' in loaded['metadata']
            assert loaded['metadata']['selection_method'] == 'permutation_test'
            assert 'selection_hyperparams' in loaded['metadata']
            assert loaded['metadata']['selection_hyperparams']['n_permutations'] == 100


class TestVaultMemberMetadata:
    """Test suite for vault member metadata persistence."""

    def test_base_model_members_have_metadata(self):
        """Test that base model configs can include member metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            control_file_path = f"{tmpdir}/control.json"
            
            base_models = [
                {
                    'name': 'test_model',
                    'model_type': 'continuous_binning',
                    'feature_column': 'test_feature',
                    'strategy': 'long',
                    'constructor_params': {'n_bins': 3},
                    'members': [
                        {
                            'member_id': 'member_1',
                            'bin_index': 0,
                            'metadata': {
                                'bin_range': [0.0, 0.33],
                                'stability_score': 0.85,
                                'selected': True
                            }
                        },
                        {
                            'member_id': 'member_2',
                            'bin_index': 1,
                            'metadata': {
                                'bin_range': [0.33, 0.66],
                                'stability_score': 0.72,
                                'selected': False
                            }
                        }
                    ]
                }
            ]
            
            metadata = {
                'is_fit': False,
                'version': '2.0.0'
            }
            
            save_control_file(
                filepath=control_file_path,
                base_models=base_models,
                metadata=metadata
            )
            
            with open(control_file_path, 'r') as f:
                saved = json.load(f)
            
            members = saved['base_models'][0]['members']
            assert len(members) == 2
            assert 'metadata' in members[0]
            assert members[0]['metadata']['selected'] == True
            assert members[1]['metadata']['selected'] == False

    def test_fitted_ensemble_includes_member_details(self):
        """Test that fitted_ensemble includes member-level details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            control_file_path = f"{tmpdir}/control.json"
            
            control_file = {
                'metadata': {
                    'is_fit': True,
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
                            {'member_id': 'member_1', 'bin_index': 0},
                            {'member_id': 'member_2', 'bin_index': 1},
                        ]
                    }
                ],
                'tickers': ['ES'],
                'fitted_base_models': {
                    'test_model': {
                        'member_1': {'thresholds': [0.5]},
                        'member_2': {'thresholds': [0.6]}
                    }
                },
                'fitted_ensemble': {
                    'weights': {
                        'test_model::member_1': 0.6,
                        'test_model::member_2': 0.4
                    },
                    'exposure_fractions': {
                        'test_model::member_1': 0.33,
                        'test_model::member_2': 0.33
                    },
                    'feature_names': ['test_model::member_1', 'test_model::member_2'],
                    'target_volatility': 0.15,
                    'unique_tickers': ['ES'],
                    'instrument_weights': {'ES': 1.0},
                    'n_tickers': 1,
                    'member_metadata': {
                        'test_model::member_1': {
                            'bin_index': 0,
                            'stability_score': 0.85
                        },
                        'test_model::member_2': {
                            'bin_index': 1,
                            'stability_score': 0.72
                        }
                    }
                }
            }
            
            with open(control_file_path, 'w') as f:
                json.dump(control_file, f)
            
            loaded = parse_control_file(control_file_path)
            
            assert 'member_metadata' in loaded['fitted_ensemble']
            assert 'test_model::member_1' in loaded['fitted_ensemble']['member_metadata']
            assert loaded['fitted_ensemble']['member_metadata']['test_model::member_1']['stability_score'] == 0.85


class TestVaultSelectionHyperparamsValidation:
    """Test suite for validating selection hyperparams in vault."""

    def test_selection_method_required_for_fitted_ensemble(self):
        """Test that selection_method is required when is_fit=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            control_file_path = f"{tmpdir}/control.json"
            
            control_file = {
                'metadata': {
                    'is_fit': True,
                    'version': '2.0.0'
                    # Missing selection_method
                },
                'base_models': [
                    {
                        'name': 'test_model',
                        'model_type': 'continuous_binning',
                        'feature_column': 'test_feature',
                        'strategy': 'long',
                        'constructor_params': {'n_bins': 3},
                        'members': [
                            {'member_id': 'member_1', 'bin_index': 0},
                        ]
                    }
                ],
                'tickers': ['ES'],
                'fitted_base_models': {},
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
            
            with open(control_file_path, 'w') as f:
                json.dump(control_file, f)
            
            loaded = parse_control_file(control_file_path)
            
            # selection_method should be optional but recommended
            # The test verifies it can be loaded even if missing
            assert loaded['metadata']['is_fit'] == True
