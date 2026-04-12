from __future__ import annotations

from pathlib import Path

from prop_firms.base.config_loader import load_account_definitions
from prop_firms.base.models import AccountDefinition


DEFAULT_TRADEDAY_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_tradeday_accounts(
    config_path: Path | None = None,
) -> dict[str, AccountDefinition]:
    """Load all TradeDay account definitions from config."""

    resolved_path = DEFAULT_TRADEDAY_CONFIG_PATH if config_path is None else config_path
    return load_account_definitions(resolved_path)


def load_tradeday_account(
    account_code: str,
    config_path: Path | None = None,
) -> AccountDefinition:
    """Load one TradeDay account definition by account code."""

    accounts = load_tradeday_accounts(config_path=config_path)
    if account_code not in accounts:
        raise KeyError(
            f"Unknown TradeDay account '{account_code}'. "
            f"Available accounts: {sorted(accounts)}"
        )
    return accounts[account_code]


def create_tradeday_simulator(
    config_path: Path | None = None,
) -> "TradeDayRuleEngine":
    """Factory for the TradeDay simulator."""

    from prop_firms.tradeday.rules import TradeDayRuleEngine

    return TradeDayRuleEngine(
        account_definitions=load_tradeday_accounts(config_path=config_path)
    )


def create_tradeday_portfolio_simulator(
    config_path: Path | None = None,
) -> "TradeDayPortfolioSimulator":
    """Factory for the TradeDay multi-account portfolio simulator."""

    from prop_firms.tradeday.portfolio_simulator import TradeDayPortfolioSimulator

    return TradeDayPortfolioSimulator(
        account_definitions=load_tradeday_accounts(config_path=config_path)
    )
