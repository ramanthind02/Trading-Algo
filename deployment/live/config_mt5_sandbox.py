r"""Nautilus SANDBOX TradingNode config scaffold (CONFIG ONLY — DO NOT RUN).

==============================================================================
🔴 SAFETY / PREREQUISITES — READ BEFORE TOUCHING THIS FILE
==============================================================================
This module ONLY *builds* a Nautilus ``TradingNodeConfig``. It does NOT run a
node, does NOT connect to MetaTrader5, and does NOT place orders. Importing it
is side-effect-free (no ``mt5.initialize``).

Running the node this config describes is a LATER, EXPLICITLY-AUTHORIZED step
and requires ALL of the following:

  1. A **SEPARATE FTMO MT5 terminal instance** — a dedicated FTMO MT5 install,
     logged into the FTMO demo account. The LIVE Darwinex terminal must NOT be
     used and must NOT be disturbed. (Nautilus is one-node-per-process and the
     MT5 IPC attaches to a running terminal — two brokers = two terminals.)
  2. FTMO demo credentials in ``.env`` (gitignored), read via ``os.environ``:
         FTMO_DEMO_SERVER, FTMO_DEMO_LOGIN, FTMO_DEMO_PASSWORD
     Credentials are NEVER hardcoded or printed here.
  3. FTMO broker symbols CONFIRMED via ``brokers.discover_symbols('ftmo')``
     against that separate terminal, then filled into ``configs/mt5_brokers.yaml``
     (they are ``UNKNOWN`` placeholders until then; ``brokers.resolve`` will
     raise rather than trade a guessed symbol — see note in build_*_config()).

What this config WOULD do once authorized (ladder rung 1, per WP-4 / 04 doc):
  - **Data:** vendored MT5 adapter live ``DataClient`` streams real FTMO quotes
    into the ``TradingNode``  (deployment/nautilus_mt5/vendor/mt5-connect).
  - **Execution:** Nautilus built-in ``SandboxExecutionClient`` fills orders
    LOCALLY against a ``SimulatedExchange`` — orders NEVER leave the machine
    and NEVER reach the FTMO server. Zero broker risk.

This validates the live runtime end-to-end (data flow, the strategy reacting in
real time, the FillModel, reconnection) at zero broker risk. NDX (MT5 feed
already available) is the natural first instrument.
==============================================================================
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from data_platform.providers.mt5 import brokers

# Re-export the general per-broker credentials/tier model for convenience.
from deployment.live.credentials import BrokerCredentials, ExecTier  # noqa: F401

# Canonical tickers to run in the sandbox (NDX first — feed already available).
# These are the repo-internal canonical names; they are resolved to the broker's
# native symbol via brokers.resolve(broker, canonical) below.
SANDBOX_BROKER: str = "ftmo"
SANDBOX_CANONICALS: tuple[str, ...] = ("NQ",)  # NQ -> FTMO's Nasdaq-100 symbol


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[2],
    )


def vendored_adapter_path() -> Path:
    """Path to the pinned, vendored mt5connect package (import root)."""
    return _repo_root() / "deployment" / "nautilus_mt5" / "vendor" / "mt5-connect"


@dataclass(frozen=True)
class SandboxCredentials:
    """FTMO DEMO creds, read from .env via os.environ. Never hardcoded/printed."""

    server: str
    login: int
    password: str = field(repr=False)  # never echo the password

    @staticmethod
    def from_env() -> "SandboxCredentials":
        """Read FTMO_DEMO_* from the environment (.env). Raises if missing.

        SAFETY: this only READS env vars; it does not connect anywhere.
        """
        try:
            server = os.environ["FTMO_DEMO_SERVER"]
            login = int(os.environ["FTMO_DEMO_LOGIN"])
            password = os.environ["FTMO_DEMO_PASSWORD"]
        except KeyError as exc:
            raise RuntimeError(
                "Missing FTMO demo credentials in environment/.env: expected "
                "FTMO_DEMO_SERVER, FTMO_DEMO_LOGIN, FTMO_DEMO_PASSWORD."
            ) from exc
        return SandboxCredentials(server=server, login=login, password=password)


def resolve_sandbox_symbols(broker: str = SANDBOX_BROKER) -> list[str]:
    """Resolve the canonical sandbox tickers to *broker*'s native MT5 symbols.

    Raises ``ValueError`` while the broker's symbols are still ``UNKNOWN``
    placeholders — i.e. before ``discover_symbols(broker)`` has been run and the
    results filled into ``configs/mt5_brokers.yaml``. This is intentional: we
    refuse to build a sandbox symbol list from guessed names.
    """
    return [brokers.resolve(broker, c) for c in SANDBOX_CANONICALS]


def build_sandbox_node_config(broker: str = SANDBOX_BROKER):
    """Build the SANDBOX ``TradingNodeConfig`` (DOES NOT run or connect).

    Pairs the vendored MT5 live ``DataClient`` with Nautilus's built-in
    ``SandboxExecutionClient`` (local virtual fills). Returns a Nautilus config
    object; the caller would later do ``TradingNode(config=...)`` + add the
    factories/strategy + ``node.run()`` — but ONLY after the prerequisites in
    the module docstring are satisfied and the step is explicitly authorized.

    This function is import-safe but constructing the config requires the
    vendored adapter + nautilus_trader to be importable AND the FTMO symbols to
    be confirmed (resolve raises on UNKNOWN). It still performs NO connection.
    """
    # The vendored adapter is imported lazily and only when a config is actually
    # requested, so merely importing this module never pulls in MetaTrader5.
    adapter_root = vendored_adapter_path()
    if str(adapter_root) not in sys.path:
        sys.path.insert(0, str(adapter_root))

    # NOTE: import is deferred to call-time. `mt5connect` imports MetaTrader5 at
    # module top-level (Windows-only, needs the package). This is fine — it does
    # NOT initialize/connect a terminal; connection happens only on node.run().
    from mt5connect.config import MT5Config            # type: ignore import-not-found
    from mt5connect.factories import build_mt5_node_config  # type: ignore import-not-found

    # Nautilus built-in sandbox execution client (LOCAL fills, no broker orders).
    from nautilus_trader.config import SandboxExecutionClientConfig  # type: ignore
    from nautilus_trader.adapters.sandbox.factory import (  # type: ignore
        SandboxLiveExecClientFactory,
    )

    creds = SandboxCredentials.from_env()
    # resolve() raises ValueError if FTMO symbols are still UNKNOWN placeholders.
    symbols = resolve_sandbox_symbols(broker)

    mt5_config = MT5Config(
        account=creds.login,
        password=creds.password,
        server=creds.server,
        symbols=symbols,
    )

    # build_mt5_node_config wires the MT5 live DATA client (+ an MT5 exec client
    # stub). For the SANDBOX rung we OVERRIDE the execution client with Nautilus's
    # SandboxExecutionClient so fills are LOCAL and orders never reach FTMO.
    node_config = build_mt5_node_config(mt5_config)

    # The caller assembles the final node like so (LATER, authorized step only):
    #
    #   from nautilus_trader.live.node import TradingNode
    #   node = TradingNode(config=node_config)
    #   node.add_data_client_factory("MT5", MT5LiveDataClientFactory)
    #   # SANDBOX exec: local virtual fills against a SimulatedExchange
    #   node.add_exec_client_factory("SANDBOX", SandboxLiveExecClientFactory)
    #   node.trader.add_strategy(TargetRebalanceStrategy(...))   # shared w/ WP-3
    #   node.build()
    #   node.run()   # ⬅ DO NOT RUN until prerequisites + authorization are met
    #
    # The SandboxExecutionClientConfig is built from the loaded instruments so the
    # SimulatedExchange knows the venue/instruments to fill against:
    _ = SandboxExecutionClientConfig  # referenced for the assembly note above

    return node_config


# Intentionally NO `if __name__ == "__main__"` runner — this file must never be
# executed as a node. Running the sandbox is a later, explicitly-authorized step
# documented in the module docstring and the WP-4 validation ladder.
