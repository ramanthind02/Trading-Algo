from datetime import datetime
from pathlib import Path

import pytest

from portfolio_research.config import (
    PortfolioResearchConfig,
    ResearchWindow,
    load_config,
)
from utils.core.enums import Ticker, TimeFrame


def test_research_window_order_validation() -> None:
    train = ResearchWindow(start=datetime(2000, 1, 1), end=datetime(2005, 12, 31))
    validation = ResearchWindow(start=datetime(2006, 1, 1), end=datetime(2010, 12, 31))
    test = ResearchWindow(start=datetime(2011, 1, 1), end=datetime(2015, 12, 31))

    config = PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2015, 12, 31),
        use_cache=True,
        train_window=train,
        validation_window=validation,
        test_window=test,
        ensemble_dirs={"dummy": "vault/D/dummy"},
    )

    assert config.train_window is train
    assert config.validation_window is validation
    assert config.test_window is test


def test_invalid_window_order_raises() -> None:
    train = ResearchWindow(start=datetime(2000, 1, 1), end=datetime(2010, 12, 31))
    validation = ResearchWindow(start=datetime(2005, 1, 1), end=datetime(2012, 12, 31))
    test = ResearchWindow(start=datetime(2013, 1, 1), end=datetime(2015, 12, 31))

    with pytest.raises(ValueError, match="train_window.end must be before"):
        PortfolioResearchConfig(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2000, 1, 1),
            end=datetime(2015, 12, 31),
            use_cache=True,
            train_window=train,
            validation_window=validation,
            test_window=test,
            ensemble_dirs={"dummy": "vault/D/dummy"},
        )


def test_discover_ensemble_dirs_uses_vault_relative_paths(tmp_path: Path, monkeypatch) -> None:
    vault_root = tmp_path / "vault"
    d_dir = vault_root / "D"
    ensemble_dir = d_dir / "example_ensemble_long"
    features_dir = ensemble_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "feature.json").write_text("{}", encoding="utf-8")

    # Point the config module's _PORTFOLIO_RESEARCH_DIR parent to our temp root
    # and import the helper directly to avoid interacting with real vault/.
    import portfolio_research.config as cfg

    monkeypatch.setattr(cfg, "_PORTFOLIO_RESEARCH_DIR", tmp_path / "portfolio_research")

    discovered = cfg._discover_ensemble_dirs()

    assert "example_ensemble_long" in discovered
    expected = str(Path("vault/D/example_ensemble_long"))
    assert discovered["example_ensemble_long"].startswith(expected)


def test_load_config_returns_portfolio_research_config() -> None:
    config = load_config()
    assert len(config.tickers) >= 1
    assert config.ensemble_dirs
    assert config.timeframe == TimeFrame.D
    assert config.start < config.end
    assert config.train_window.start >= config.start
    assert config.test_window.end <= config.end


def test_portfolio_research_config_empty_ensemble_dirs_raises() -> None:
    with pytest.raises(ValueError, match="ensemble_dirs must be non-empty"):
        PortfolioResearchConfig(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2000, 1, 1),
            end=datetime(2023, 12, 31),
            use_cache=True,
            train_window=ResearchWindow(
                start=datetime(2000, 1, 1),
                end=datetime(2005, 12, 31),
            ),
            validation_window=ResearchWindow(
                start=datetime(2006, 1, 1),
                end=datetime(2010, 12, 31),
            ),
            test_window=ResearchWindow(
                start=datetime(2011, 1, 1),
                end=datetime(2015, 12, 31),
            ),
            ensemble_dirs={},
        )


def test_portfolio_research_config_invalid_baseline_mode_raises() -> None:
    with pytest.raises(ValueError, match="baseline_mode must be"):
        PortfolioResearchConfig(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2000, 1, 1),
            end=datetime(2023, 12, 31),
            use_cache=True,
            train_window=ResearchWindow(
                start=datetime(2000, 1, 1),
                end=datetime(2005, 12, 31),
            ),
            validation_window=ResearchWindow(
                start=datetime(2006, 1, 1),
                end=datetime(2010, 12, 31),
            ),
            test_window=ResearchWindow(
                start=datetime(2011, 1, 1),
                end=datetime(2015, 12, 31),
            ),
            ensemble_dirs={"a": "vault/D/some_ensemble"},
            baseline_mode="invalid",
        )


def test_portfolio_research_config_rejects_removed_sector_surface() -> None:
    with pytest.raises(TypeError):
        PortfolioResearchConfig(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2000, 1, 1),
            end=datetime(2023, 12, 31),
            use_cache=True,
            train_window=ResearchWindow(
                start=datetime(2000, 1, 1),
                end=datetime(2005, 12, 31),
            ),
            validation_window=ResearchWindow(
                start=datetime(2006, 1, 1),
                end=datetime(2010, 12, 31),
            ),
            test_window=ResearchWindow(
                start=datetime(2011, 1, 1),
                end=datetime(2015, 12, 31),
            ),
            ensemble_dirs={"a": "vault/D/some_ensemble"},
            sector_allocation_config_path="feature_research/config/sector.json",  # type: ignore[call-arg]
        )
