"""
Unit tests for ensemble bugfixes (return types, validation, factory, vault).

Covers: uniform_binning rejection, WeightLayer fdm_max propagation,
validate_control_file strictness, load_feature_base_models raise on missing file.
"""

import tempfile
from pathlib import Path

import pytest

from ensemble.ensemble_utils import validate_base_model_config
from ensemble.weight_layer import WeightLayer, WeightLayerConfig


class TestValidateBaseModelConfig:
    """Tests for validate_base_model_config behavior."""

    def test_uniform_binning_rejected(self) -> None:
        """Config with model_type='uniform_binning' raises at validation (not at factory)."""
        config = {
            "name": "test",
            "model_type": "uniform_binning",
            "feature_column": "rsi_signal_D",
            "strategy": "long",
            "constructor_params": {"n_bins": 5},
            "members": [{"member_name": "m1", "params": {"n_bins": 5}}],
        }
        with pytest.raises(ValueError) as exc_info:
            validate_base_model_config(config, index=0)
        assert "uniform_binning" in str(exc_info.value).lower()
        assert "not supported" in str(exc_info.value).lower() or "continuous_binning" in str(exc_info.value)


class TestWeightLayerFactory:
    """Tests for WeightLayer factory fdm_max and defaults."""

    def test_fdm_max_used_for_equal_signal(self) -> None:
        """WeightLayer(weight_method='equal_signal', fdm_max=1.5) results in fdm_max=1.5."""
        layer = WeightLayer(weight_method="equal_signal", fdm_max=1.5)
        assert layer.fdm_max == 1.5

    def test_fdm_max_default_2(self) -> None:
        """Default fdm_max is 2.0 (per spec)."""
        layer = WeightLayer(weight_method="equal_signal")
        assert layer.fdm_max == 2.0

    def test_config_overrides_fdm_max_arg(self) -> None:
        """When config is provided, config.fdm_max is used."""
        config = WeightLayerConfig(weighting_method="equal_signal", fdm_max=1.8)
        layer = WeightLayer(config=config, fdm_max=99.0)
        assert layer.fdm_max == 1.8


class TestControlFileValidation:
    """Tests for validate_control_file truthiness."""

    def test_is_fit_false_rejects_any_fitted_base_models_key(self) -> None:
        """When is_fit=False, presence of 'fitted_base_models' key raises."""
        from ensemble.ensemble_utils import validate_control_file

        control = {
            "metadata": {"is_fit": False},
            "base_models": [],
            "fitted_base_models": {},
        }
        with pytest.raises(ValueError) as exc_info:
            validate_control_file(control)
        assert "fitted_base_models" in str(exc_info.value)

    def test_is_fit_false_rejects_any_fitted_ensemble_key(self) -> None:
        """When is_fit=False, presence of 'fitted_ensemble' key raises."""
        from ensemble.ensemble_utils import validate_control_file

        control = {
            "metadata": {"is_fit": False},
            "base_models": [],
            "fitted_ensemble": None,
        }
        with pytest.raises(ValueError) as exc_info:
            validate_control_file(control)
        assert "fitted_ensemble" in str(exc_info.value)


class TestLoadFeatureBaseModelsRaises:
    """load_feature_base_models raises on missing feature file."""

    def test_load_feature_base_models_raises_on_missing_file(self) -> None:
        """Missing feature file raises FileNotFoundError (no silent {})."""
        from ensemble.vault_manager import load_feature_base_models

        with tempfile.TemporaryDirectory() as tmp:
            # Empty dir: no features/ subdir or file
            ensemble_dir = Path(tmp) / "D" / "test_long"
            ensemble_dir.mkdir(parents=True)
            (ensemble_dir / "features").mkdir(parents=True)
            # No feature file for 'nonexistent_feature'
            with pytest.raises(FileNotFoundError) as exc_info:
                load_feature_base_models(
                    "nonexistent_feature",
                    ensemble_dir=str(ensemble_dir),
                )
            assert "nonexistent_feature" in str(exc_info.value)
            assert "not found" in str(exc_info.value).lower()
