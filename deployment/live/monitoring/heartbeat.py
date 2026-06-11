"""Periodic portfolio heartbeat actor — Nautilus-native monitoring.

A lightweight :class:`Actor` that logs a one-line health snapshot on a timer:
account equity, open net positions, and (optionally) the warmup state of the
vault forecast engine. It reads everything from the Nautilus cache/portfolio, so
it adds no broker traffic of its own. Logs go through the Nautilus logger, so
they land in the same structured stdout/file stream as the rest of the node.
"""
from __future__ import annotations

from datetime import timedelta

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency

_HEARTBEAT_TIMER = "vault-heartbeat"


class PortfolioHeartbeatConfig(ActorConfig, frozen=True):
    venue: str = "MT5"
    account_currency: str = "USD"
    interval_secs: int = 300


class PortfolioHeartbeat(Actor):
    def __init__(self, config: PortfolioHeartbeatConfig) -> None:
        super().__init__(config)
        self._venue = Venue(config.venue)
        self._currency = Currency.from_str(config.account_currency)

    def on_start(self) -> None:
        self.clock.set_timer(
            name=_HEARTBEAT_TIMER,
            interval=timedelta(seconds=self.config.interval_secs),
            callback=self._beat,
        )
        self._beat(None)  # emit one immediately so startup state is visible

    def on_stop(self) -> None:
        if _HEARTBEAT_TIMER in self.clock.timer_names:
            self.clock.cancel_timer(_HEARTBEAT_TIMER)

    def _beat(self, _event) -> None:
        equity = self._equity_str()
        positions = self.cache.positions_open(venue=self._venue)
        if positions:
            legs = ", ".join(
                f"{p.instrument_id.symbol}={p.signed_qty}" for p in positions
            )
        else:
            legs = "flat"
        self.log.info(f"[heartbeat] equity={equity} positions: {legs}")

    def _equity_str(self) -> str:
        account = self.cache.account_for_venue(self._venue)
        if account is None:
            return "n/a"
        money = account.balance_total(self._currency)
        return f"{money.as_double():,.2f} {self._currency.code}" if money is not None else "n/a"


__all__ = ["PortfolioHeartbeat", "PortfolioHeartbeatConfig"]
