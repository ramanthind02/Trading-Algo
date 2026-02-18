from datetime import datetime
from pathlib import Path

from feature_research.rule_based.config import RuleBasedResearchConfig, load_config
from utils.enums import Ticker


def test_load_config_returns_rule_based_research_config():
    config = load_config()
    assert isinstance(config, RuleBasedResearchConfig)


def test_load_config_defaults():
    config = load_config()
    assert Ticker.ES in config.tickers
    assert Ticker.NQ in config.tickers
    assert config.start == datetime(2020, 1, 1)
    assert config.end == datetime(2024, 12, 31)
    assert config.bias_spec["module_name"] == "rsi_signal"
    assert isinstance(config.bias_spec["params"]["rsi_period"], list)
    assert config.target_col == "log_return"
    assert config.use_cache is True
    assert config.populate_cache is True


def test_reports_dir_includes_module_name():
    config = load_config()
    assert config.bias_spec["module_name"] in str(config.reports_dir)
    assert "rule_based" in str(config.reports_dir)
