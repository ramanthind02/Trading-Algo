r"""Assemble the live vault TradingNode (sandbox execution tier).

Pairs the vendored MT5 **live data client** (real broker quotes) with Nautilus's
built-in **SandboxExecutionClient** (local virtual fills against a
``SimulatedExchange`` — orders NEVER reach the broker), both on the same ``MT5``
venue so the instruments loaded by the MT5 provider are the ones the sandbox
fills against. Adds the :class:`VaultRebalanceStrategy` and the
:class:`PortfolioHeartbeat` actor, plus Nautilus structured logging.

Building this module is side-effect-free. ``build_sandbox_node(..., connect=False)``
returns an *unbuilt* node (validates wiring without touching a terminal);
``connect=True`` calls ``node.build()`` which connects to the bound MT5 terminal
and loads instruments. ``node.run()`` is the caller's explicit, armed step.

Safety: this targets a SEPARATE broker terminal (FTMO demo) via ``MT5Config.path``
— the live Darwinex terminal must not be used. Credentials are read from the
environment only.
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from data_platform.providers.mt5 import brokers
from deployment.live.broker_data import broker_cache_root
from deployment.live.config_mt5_sandbox import vendored_adapter_path
from deployment.live.monitoring.live_state import live_state_dir
from deployment.live.credentials import BrokerCredentials, ExecTier
from deployment.live.monitoring.heartbeat import PortfolioHeartbeat, PortfolioHeartbeatConfig
from deployment.live.vault_strategy import VaultRebalanceConfig, VaultRebalanceStrategy


@dataclass(frozen=True)
class LiveRuntimeConfig:
    """Runtime knobs for the live vault node (loaded from JSON)."""

    broker: str = "ftmo"
    venue: str = "MT5"
    # Signal source of truth — every execution broker generates signals from this
    # single feed (Darwinex). Execution stays per-broker; only the signal cache is shared.
    signal_broker: str = "darwinex"
    vault_root: str = "vault"
    tickers: tuple[str, ...] = ("ES", "NQ", "GC", "SI")
    target_volatility: float = 0.15
    max_position_pct: float = 2.5
    idm_max: float = 2.5
    warmup_min_bars: int = 500
    prediction_daily_max_bars: int = 500
    poll_interval_secs: int = 20
    quote_window_timeout_secs: int = 90
    # Rollover overlay knobs (broker minutes):
    exit_lead_min: int = 15
    entry_settle_min: int = 0
    entry_settle_metal_min: int = 5
    cross_margin: float = 1.0
    sizing_basis_usd: float = 0.0          # 0 = read sandbox account equity
    account_currency: str = "USD"
    lot_size_ceiling: float = 100.0
    min_rebalance_lots: float = 0.01
    min_rebalance_notional_usd: float = 50.0
    sandbox_starting_balance: float = 100_000.0
    initial_balance: float = 0.0  # prop-firm total-loss anchor (0 = first observed equity)
    heartbeat_secs: int = 300
    log_level: str = "INFO"
    # Execution tier: "sandbox" (local fills, default) | "demo" | "live" (real orders).
    exec_tier: str = "sandbox"
    # Robustness: withhold the first decision this many seconds after start so broker
    # position/account reconciliation can populate (so a crash+restart never trades
    # against a phantom-flat book). 0 = no wait (sandbox). A stale quote older than
    # max_quote_age_secs defers the symbol rather than filling into a dead book.
    startup_grace_secs: int = 60
    max_quote_age_secs: float = 600.0
    log_directory: str = "logs/live_nautilus"

    @classmethod
    def from_json(cls, path: str | Path) -> "LiveRuntimeConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # tolerate a "_comment" key and unknown extras; coerce tickers to tuple
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in raw.items() if k in fields}
        if "tickers" in kwargs:
            kwargs["tickers"] = tuple(kwargs["tickers"])
        return cls(**kwargs)


def _strategy_config(
    runtime: LiveRuntimeConfig,
    credentials: BrokerCredentials | None = None,
    exec_tier: ExecTier = ExecTier.SANDBOX,
) -> VaultRebalanceConfig:
    return VaultRebalanceConfig(
        broker=runtime.broker,
        venue=runtime.venue,
        vault_root=runtime.vault_root,
        # Signals read the single shared Darwinex cache (signal_broker), NOT the
        # execution broker's own series — every venue trades one identical signal.
        cache_root=str(broker_cache_root(runtime.signal_broker)),
        tickers=runtime.tickers,
        target_volatility=runtime.target_volatility,
        max_position_pct=runtime.max_position_pct,
        idm_max=runtime.idm_max,
        warmup_min_bars=runtime.warmup_min_bars,
        prediction_daily_max_bars=runtime.prediction_daily_max_bars,
        poll_interval_secs=runtime.poll_interval_secs,
        quote_window_timeout_secs=runtime.quote_window_timeout_secs,
        exit_lead_min=runtime.exit_lead_min,
        entry_settle_min=runtime.entry_settle_min,
        entry_settle_metal_min=runtime.entry_settle_metal_min,
        cross_margin=runtime.cross_margin,
        sizing_basis_usd=runtime.sizing_basis_usd,
        account_currency=runtime.account_currency,
        lot_size_ceiling=runtime.lot_size_ceiling,
        min_rebalance_lots=runtime.min_rebalance_lots,
        min_rebalance_notional_usd=runtime.min_rebalance_notional_usd,
        startup_grace_secs=runtime.startup_grace_secs,
        max_quote_age_secs=runtime.max_quote_age_secs,
        # ── live-state publication (node-mediated dashboard feed) ─────────
        live_state_dir=str(live_state_dir(runtime.broker)),
        exec_tier=exec_tier.value,
        account_login=credentials.login if credentials is not None else 0,
        account_server=credentials.server if credentials is not None else "",
        initial_balance=runtime.initial_balance,
    )


def _resolve_native_symbols(runtime: LiveRuntimeConfig) -> list[str]:
    natives: list[str] = []
    for canonical in runtime.tickers:
        try:
            natives.append(brokers.resolve(runtime.broker, canonical))
        except (ValueError, KeyError):
            # Unresolvable on this broker (e.g. UNKNOWN placeholder) — skip; the
            # strategy logs and trades only what resolves.
            continue
    if not natives:
        raise ValueError(
            f"No tickers in {runtime.tickers} resolve on broker {runtime.broker!r}"
        )
    return natives


def build_node(
    runtime: LiveRuntimeConfig,
    credentials: BrokerCredentials,
    *,
    connect: bool,
    exec_tier: ExecTier = ExecTier.SANDBOX,
):
    """Build the live TradingNode for ``exec_tier``.

    All tiers stream the broker's live DATA via the vendored MT5 data client. The
    execution client differs:

    * ``SANDBOX`` — Nautilus ``SandboxExecutionClient`` (NETTING): fills are LOCAL,
      orders never reach the broker. Zero broker risk.
    * ``DEMO`` / ``LIVE`` — vendored ``MT5LiveExecClientFactory``: orders are sent
      to the broker (demo or funded account), tagged with the broker's magic
      number from ``brokers.execution_rules``.

    ``credentials`` must be for ``runtime.broker`` (real for an armed run; dummy is
    acceptable only for ``connect=False`` wiring validation). ``connect=True`` calls
    ``node.build()`` (connects + loads instruments); ``False`` returns it unbuilt.
    """
    import msgspec

    adapter_root = vendored_adapter_path()
    if str(adapter_root) not in sys.path:
        sys.path.insert(0, str(adapter_root))

    from mt5connect.config import MT5Config  # type: ignore import-not-found
    from mt5connect.factories import (  # type: ignore import-not-found
        MT5LiveDataClientFactory,
        MT5LiveExecClientFactory,
    )
    from mt5connect.factories import build_mt5_node_config  # type: ignore import-not-found
    from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
    from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.live.node import TradingNode

    symbols = _resolve_native_symbols(runtime)
    from deployment.live.monitoring.live_state import live_state_dir

    mt5_config = MT5Config(
        account=credentials.login,
        password=credentials.password,
        server=credentials.server,
        symbols=symbols,
        path=brokers.terminal_path(runtime.broker),
        magic_number=brokers.execution_rules(runtime.broker).magic_number,
        # Durable full-field deal capture (migration plan §7.1): the exec
        # client appends every magic-tagged deal here; `registry ingest live`
        # loads them. Capture failures never reach the trading loop.
        deals_log_path=str(Path(live_state_dir(runtime.broker)) / "deals.jsonl"),
        # Durable order_send response capture (migration plan §7.2): appends
        # every broker response (retcode, deal, order, volume, price, bid, ask)
        # for both successes and rejections. Capture failures never reach the loop.
        submit_results_log_path=str(
            Path(live_state_dir(runtime.broker)) / "submit_results.jsonl"
        ),
    )

    # Per-broker log stream: logs/live_nautilus/{broker}_{date}.log + stdout.
    logging_cfg = LoggingConfig(
        log_level=runtime.log_level,
        log_level_file=runtime.log_level,
        log_directory=runtime.log_directory,
        log_file_name=runtime.broker,
    )
    node_config = build_mt5_node_config(mt5_config, logging_config=logging_cfg)

    if exec_tier is ExecTier.SANDBOX:
        # Replace the MT5 exec client with the local sandbox (NETTING) on the venue.
        sandbox_exec = SandboxExecutionClientConfig(
            venue=runtime.venue,
            starting_balances=[f"{int(runtime.sandbox_starting_balance)} {runtime.account_currency}"],
            base_currency=runtime.account_currency,
            oms_type="NETTING",
        )
        node_config = msgspec.structs.replace(node_config, exec_clients={runtime.venue: sandbox_exec})
        exec_factory = SandboxLiveExecClientFactory
    else:
        # demo / live: keep the MT5 exec client wired by build_mt5_node_config (REAL orders).
        # H8 (FIXED 2026-06): on a HEDGING account a plain opposing DEAL used to open a NEW
        # ticket (a hedge) rather than reduce the position. The exec client now NETS
        # per-ticket — MT5LiveExecutionClient._submit_market_netting closes opposing
        # magic tickets (position=ticket, FIFO) and opens a residual only on a true
        # reversal, so the no-hedge rule is honoured. FTMO `.cash` legs and live fills are
        # also tracked now (the same 2026-06 adapter fixes). Still demo-validate first.
        warnings.warn(
            f"build_node: exec_tier={exec_tier.value} sends REAL orders to {runtime.broker}. "
            "The exec client nets on HEDGING accounts (closes opposing tickets per-ticket, "
            "never hedges) after the 2026-06 adapter fixes — verify net exposure on demo "
            "before arming a funded account.",
            stacklevel=2,
        )
        exec_factory = MT5LiveExecClientFactory

    node = TradingNode(config=node_config)
    node.add_data_client_factory(runtime.venue, MT5LiveDataClientFactory)
    node.add_exec_client_factory(runtime.venue, exec_factory)
    node.trader.add_strategy(
        VaultRebalanceStrategy(_strategy_config(runtime, credentials, exec_tier))
    )
    node.trader.add_actor(
        PortfolioHeartbeat(
            PortfolioHeartbeatConfig(
                venue=runtime.venue,
                account_currency=runtime.account_currency,
                interval_secs=runtime.heartbeat_secs,
            )
        )
    )
    if connect:
        node.build()
    return node


def build_sandbox_node(runtime: LiveRuntimeConfig, credentials, *, connect: bool):
    """Back-compat wrapper: build the SANDBOX node (local fills)."""
    return build_node(runtime, credentials, connect=connect, exec_tier=ExecTier.SANDBOX)


__all__ = ["LiveRuntimeConfig", "build_node", "build_sandbox_node"]
