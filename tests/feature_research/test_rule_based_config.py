from datetime import datetime
from pathlib import Path

from feature_research.in_sample.rule_based.config import RuleBasedResearchConfig, load_config
from feature_research.walkforward.config import WalkforwardResearchConfig
from utils.enums import Ticker


def test_load_config_returns_rule_based_research_config():
    config = load_config()
    assert isinstance(config, RuleBasedResearchConfig)


def test_load_config_defaults():
    config = load_config()
    assert Ticker.ES in config.tickers
    assert Ticker.NQ in config.tickers
    assert config.start == datetime(2000, 1, 1)
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


def test_load_config_includes_walkforward_defaults() -> None:
    config = load_config()

    assert isinstance(config.walkforward, WalkforwardResearchConfig)
    assert config.walkforward.enabled is False
    assert config.walkforward.train_start == config.start
    assert config.walkforward.train_end == datetime(2019, 6, 25)
    assert config.walkforward.output_root == Path("feature_research/shared_results")
