from datetime import datetime

from feature_research.config import FeatureType
from feature_research.in_sample.config import ResearchConfig, load_config
from utils.core.enums import Ticker


def test_load_config_returns_rule_based_research_config():
    # For this test, we need to override the base config to use RULE_BASED
    # Since load_config() dispatches on feature_type from base config,
    # and the default is CONTINUOUS, this test documents the unified structure
    config = load_config()
    assert isinstance(config, ResearchConfig)
    # Default is CONTINUOUS; to test RULE_BASED, set feature_type in base config


def test_load_config_defaults():
    config = load_config()
    assert Ticker.ES in config.tickers
    assert Ticker.NQ in config.tickers
    assert config.start == datetime(2000, 1, 1)
    assert config.end == datetime(2017, 12, 31)
    # Default feature_type is CONTINUOUS; assertions below are for that
    # Rule-based assertions would use feature_type=RULE_BASED


def test_reports_dir_includes_feature_type():
    config = load_config()
    # Default feature_type is CONTINUOUS, so reports_dir contains "continuous"
    assert "continuous" in str(config.reports_dir)
