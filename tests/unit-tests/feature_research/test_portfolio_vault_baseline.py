from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from feature_research.config import PortfolioSourceConfig, discover_portfolio_source_ensemble_dirs, load_config
from portfolio_research.config import discover_ensemble_dirs
from utils.core.enums import Ticker, TimeFrame
from utils.vault_paths import vault_discovery_dirnames_for_profile


def test_discover_ensemble_dirs_prop_profile_excludes_personal_vault(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prop_root = tmp_path / "vault" / "D" / "momentum" / "prop_only_long" / "features"
    personal_root = tmp_path / "vault_personal" / "D" / "momentum_gc" / "personal_only_long" / "features"
    prop_root.mkdir(parents=True, exist_ok=True)
    personal_root.mkdir(parents=True, exist_ok=True)
    (prop_root / "feature.json").write_text("{}", encoding="utf-8")
    (personal_root / "feature.json").write_text("{}", encoding="utf-8")

    import portfolio_research.config as cfg

    monkeypatch.setattr(cfg, "_PORTFOLIO_RESEARCH_DIR", tmp_path / "portfolio_research")

    prop_only = discover_ensemble_dirs(
        allowed_timeframes=(TimeFrame.D,),
        vault_discovery_dirnames=vault_discovery_dirnames_for_profile("prop"),
    )
    assert set(prop_only) == {"prop_only_long"}
    assert all(path.startswith("vault/") for path in prop_only.values())


def test_discover_portfolio_source_ensemble_dirs_uses_configured_profile() -> None:
    config = load_config()
    source = replace(config.portfolio_source, vault_profile="prop", tickers=(Ticker.ES,))
    assert source is not None
    discovered = discover_portfolio_source_ensemble_dirs(source, portfolio_tickers=(Ticker.ES,))
    assert discovered
    assert all(path.startswith("vault/") for path in discovered.values())
    assert "vault_personal/" not in "".join(discovered.values())


def test_portfolio_source_explicit_ensemble_dirs_bypasses_discovery() -> None:
    source = PortfolioSourceConfig(
        ensemble_dirs={"custom": "vault/D/momentum/custom_long"},
        vault_profile="personal",
    )
    discovered = discover_portfolio_source_ensemble_dirs(source, portfolio_tickers=(Ticker.ES,))
    assert discovered == {"custom": "vault/D/momentum/custom_long"}
