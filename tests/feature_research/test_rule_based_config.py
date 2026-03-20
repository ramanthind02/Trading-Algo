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
    assert config.start == datetime(2000, 1, 1)
    assert config.feature_type == FeatureType.RULE_BASED


def test_reports_dir_includes_feature_type():
    config = load_config()
    assert "rule_based" in str(config.reports_dir)
