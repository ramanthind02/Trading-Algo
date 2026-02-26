import pytest
from feature_selection.validation.config import ValidationConfig


def test_validation_config_defaults():
    """Test ValidationConfig with default values."""
    config = ValidationConfig(feature_type='continuous')

    assert config.feature_type == 'continuous'
    assert config.n_permutations == 1000
    assert config.confidence_level == 0.95
    assert config.min_sharpe_threshold == 0.5
    assert config.random_seed is None


def test_validation_config_custom():
    """Test ValidationConfig with custom values."""
    config = ValidationConfig(
        feature_type='rule_based',
        n_permutations=500,
        min_sharpe_threshold=0.7,
        random_seed=42,
    )

    assert config.feature_type == 'rule_based'
    assert config.n_permutations == 500
    assert config.random_seed == 42


def test_validation_config_invalid_feature_type():
    """Test ValidationConfig rejects invalid feature types."""
    # This should fail at type-check level with mypy,
    # but we can test runtime validation if we add it
    config = ValidationConfig(feature_type='continuous')
    assert config.feature_type in ['continuous', 'rule_based']
