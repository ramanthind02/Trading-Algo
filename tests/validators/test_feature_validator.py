import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from feature_selection.validators.validator import FeatureValidator
from feature_selection.validators.config import ValidationConfig


@pytest.fixture
def sample_data():
    """Create sample feature and target data."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(500), name='test_feature')
    target = pd.Series(0.2 * feature + np.random.randn(500) * 0.5, name='target')
    return feature, target


@pytest.fixture
def validator(tmp_path):
    """Create FeatureValidator instance."""
    config = ValidationConfig(feature_type='continuous')

    # Minimal dependencies for now
    permutation_engine = None
    parameter_analyzer = None

    return FeatureValidator(
        config=config,
        permutation_engine=permutation_engine,
        parameter_analyzer=parameter_analyzer,
        output_dir=tmp_path,
    )


def test_validator_initialization(validator):
    """Test FeatureValidator initialization."""
    assert validator.config.feature_type == 'continuous'
    assert validator.output_dir.exists()


def test_run_eda_continuous(validator, sample_data):
    """Test run_eda for continuous feature."""
    feature, target = sample_data

    eda_report = validator.run_eda(
        feature_data=feature.to_frame(),
        target=target,
    )

    assert eda_report.feature_stats is not None
    assert eda_report.target_stats is not None
    assert 'pearson' in eda_report.correlations
    assert eda_report.adf_test is not None
    assert eda_report.kpss_test is not None
    assert eda_report.continuous_report is not None
    assert eda_report.rule_report is None


def test_run_stage1_permutation(validator, sample_data):
    """Test run_stage1_permutation method."""
    feature, target = sample_data

    # Run Stage 1 permutation test
    perm_report = validator.run_stage1_permutation(
        feature_data=feature.to_frame(),
        target=target,
    )

    assert perm_report.stage == 'stage1_vector_shuffle'
    assert perm_report.n_permutations == validator.config.n_permutations
    assert perm_report.observed_sharpe != 0
    assert len(perm_report.permuted_sharpes) == validator.config.n_permutations
    assert hasattr(perm_report, 'p_value')
    assert hasattr(perm_report, 'passed')


def test_run_full_validation(validator, sample_data):
    """Test run_full_validation method."""
    feature, target = sample_data

    # Run full validation pipeline
    validation_report = validator.run_full_validation(
        feature_data=feature.to_frame(),
        target=target,
        feature_name='test_feature',
    )

    # Check ValidationReport structure
    assert validation_report.feature_name == 'test_feature'
    assert validation_report.feature_type == 'continuous'
    assert validation_report.eda_report is not None
    assert validation_report.stage1_report is not None
    assert validation_report.validation_status in ['passed', 'failed', 'incomplete']
