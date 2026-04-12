from __future__ import annotations

from pathlib import Path

from prop_firms.base.config_loader import load_account_definitions
from prop_firms.base.models import AccountDefinition


DEFAULT_TRADEFUNDRR_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_tradefundrr_accounts(
    config_path: Path | None = None,
) -> dict[str, AccountDefinition]:
    """Load all Tradefundrr account definitions from config."""

    resolved_path = DEFAULT_TRADEFUNDRR_CONFIG_PATH if config_path is None else config_path
    return load_account_definitions(resolved_path)


def load_tradefundrr_account(
    account_code: str,
    config_path: Path | None = None,
) -> AccountDefinition:
    """Load one Tradefundrr account definition by account code."""

    accounts = load_tradefundrr_accounts(config_path=config_path)
    if account_code not in accounts:
        raise KeyError(
            f"Unknown Tradefundrr account '{account_code}'. "
            f"Available accounts: {sorted(accounts)}"
        )
    return accounts[account_code]


def create_tradefundrr_simulator(
    config_path: Path | None = None,
) -> "TradefundrrRuleEngine":
    """Factory for the Tradefundrr simulator."""

    from prop_firms.tradefundrr.rules import TradefundrrRuleEngine

    return TradefundrrRuleEngine(
        account_definitions=load_tradefundrr_accounts(config_path=config_path)
    )


def create_tradefundrr_portfolio_simulator(
    config_path: Path | None = None,
) -> "TradefundrrPortfolioSimulator":
    """Factory for the Tradefundrr multi-account portfolio simulator."""

    from prop_firms.tradefundrr.portfolio_simulator import TradefundrrPortfolioSimulator

    return TradefundrrPortfolioSimulator(
        account_definitions=load_tradefundrr_accounts(config_path=config_path)
    )
