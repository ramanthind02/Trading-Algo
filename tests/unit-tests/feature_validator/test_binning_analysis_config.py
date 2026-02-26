import pytest

from feature_research.config import BinningAnalysisConfig
from feature_research.in_sample.config import load_config


def test_binning_analysis_config_fields() -> None:
    config = load_config()
    assert isinstance(config.binning_params.bin_counts, list)
    assert all(isinstance(n, int) for n in config.binning_params.bin_counts)
    assert all(n > 0 for n in config.binning_params.bin_counts)
    assert config.binning_params.metric_threshold >= 0.0
    assert config.binning_params.use_coverage_bonus in [True, False]
    assert config.binning_params.coverage_bonus_per_10pct >= 0.0
    assert config.binning_params.max_coverage_bonus >= 0.0
    assert config.binning_params.bin_index_min >= 0
    assert config.binning_params.bin_index_max is None or config.binning_params.bin_index_max >= config.binning_params.bin_index_min


def test_binning_analysis_config_bin_index_range_defaults() -> None:
    """Default bin index range is no restriction (all bins)."""
    cfg = BinningAnalysisConfig()
    assert cfg.bin_index_min == 0
    assert cfg.bin_index_max is None


def test_binning_analysis_config_bin_index_range_accepts_explicit() -> None:
    """Explicit bin_index_min and bin_index_max are accepted."""
    cfg = BinningAnalysisConfig(bin_index_min=0, bin_index_max=3)
    assert cfg.bin_index_min == 0
    assert cfg.bin_index_max == 3


def test_binning_analysis_config_bin_index_min_negative_raises() -> None:
    """bin_index_min < 0 raises ValueError."""
    with pytest.raises(ValueError, match="bin_index_min must be >= 0"):
        BinningAnalysisConfig(bin_index_min=-1)


def test_binning_analysis_config_bin_index_max_less_than_min_raises() -> None:
    """bin_index_max < bin_index_min raises ValueError."""
    with pytest.raises(ValueError, match="bin_index_max must be >= bin_index_min"):
        BinningAnalysisConfig(bin_index_min=3, bin_index_max=2)
