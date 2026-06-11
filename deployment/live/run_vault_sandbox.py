r"""Operator entrypoint: run the vault portfolio live on MT5.

Execution tiers (``--exec-tier``, default ``sandbox``):
  * ``sandbox`` — real broker quotes stream in; orders fill LOCALLY via Nautilus's
    SandboxExecutionClient and NEVER reach the broker (zero broker risk).
  * ``demo`` / ``live`` — REAL orders to the broker's demo / funded account via the
    vendored MT5 exec client. Requires the explicit ``--live`` flag.

Usage (PowerShell, repo root)::

    # Validate wiring only — builds the node WITHOUT connecting to a terminal.
    .\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker ftmo --dry-build
    .\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker ftmo --exec-tier demo --dry-build

    # Armed SANDBOX run (paper; no broker orders):
    .\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker ftmo --arm --minutes 10

    # Armed DEMO run (REAL orders to the FTMO demo account) — needs --live:
    .\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker ftmo --exec-tier demo --arm --live

Safety:
  * credentials + terminal are resolved from the SAME ``--broker`` (no cross-binding).
  * ``darwinex`` is refused — its only terminal is the live, do-not-disturb scraper.
  * demo/live tiers require ``--live``; demo additionally asserts a DEMO server.

Importing this module is side-effect-free; nothing connects until ``--arm``.
"""
from __future__ import annotations

import argparse
import dataclasses
import signal
import sys
import threading
from pathlib import Path

try:
    from lib.core.repo_bootstrap import ensure_project_root_on_path
except ImportError:  # pragma: no cover - fallback when scripts pkg not importable
    def ensure_project_root_on_path() -> None:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

ensure_project_root_on_path()

# Standard process preamble: UTF-8 console (so the ✓/⚠️ status prints don't crash a
# cp1252 Windows console) + load the gitignored .env so broker creds (FTMO_DEMO_*,
# DARWINEX_DEMO_*, …) reach os.environ before BrokerCredentials.from_env reads them.
# .env loading falls back to a manual parser when python-dotenv is absent and never
# clobbers a pre-set shell var.
from lib.core.runtime_bootstrap import bootstrap_runtime  # noqa: E402

bootstrap_runtime()

from deployment.live.broker_data import (  # noqa: E402
    bind_signal_cache,
    refresh_signal_daily,
)
from deployment.live.credentials import BrokerCredentials, ExecTier  # noqa: E402
from deployment.live.forecast_engine import ForecastEngineConfig, VaultForecastEngine  # noqa: E402
from deployment.live.runtime.node_builder import LiveRuntimeConfig, build_node  # noqa: E402

_DEFAULT_CONFIG = "configs/live_nautilus_ftmo.json"


def _load_credentials(broker: str, tier: ExecTier, *, allow_dummy: bool) -> BrokerCredentials:
    """Resolve the TARGET broker's creds; allow a placeholder only for dry-build.

    A non-armable broker (e.g. ``darwinex``) always raises — even in dry-build —
    because there is no safe account to bind. Merely-missing creds are tolerated
    for ``--dry-build`` so wiring can be validated offline.
    """
    try:
        return BrokerCredentials.from_env(broker, tier)
    except RuntimeError as exc:
        if not allow_dummy:
            raise
        BrokerCredentials.env_prefix(broker, tier)  # re-raises for non-armable brokers
        print(f"[dry-build] {broker} {tier.value} creds not set — placeholder (no connection).")
        return BrokerCredentials(broker=broker, server="demo-placeholder", login=1, password="placeholder")


def _assert_armable(tier: ExecTier, creds: BrokerCredentials, *, live_flag: bool) -> None:
    """Guard an armed run. Sandbox is always allowed; real tiers are gated."""
    if tier is ExecTier.SANDBOX:
        return  # no broker orders — safe
    if not live_flag:
        sys.exit(
            f"ABORT: exec_tier={tier.value} places REAL orders on {creds.broker}. "
            "Pass --live to arm it (default execution is sandbox)."
        )
    if tier is ExecTier.DEMO and not creds.is_demo_server:
        sys.exit(
            f"ABORT: demo tier requires a DEMO server, got {creds.server!r} for {creds.broker}."
        )


def _refresh_signal_data(runtime: LiveRuntimeConfig) -> None:
    """Populate the shared Darwinex signal cache before the node starts.

    Broker-independent: bind the single shared signal namespace and rebuild it
    from the scraped Darwinex CFD series (``data/mt5_data`` via cfd_candles). This
    needs NO MT5 connection — the execution terminal is connected later by
    ``build_node``. Freshness depends on the daily Darwinex scraper having run.
    """
    engine = VaultForecastEngine(ForecastEngineConfig(vault_root=runtime.vault_root)).load()
    required = engine.required_tickers
    print(f"Required tickers: {list(required)}")

    store = bind_signal_cache()  # singleton -> shared Darwinex namespace
    report = refresh_signal_daily(
        required, store=store, vault_root=runtime.vault_root, populate_bias=True,
    )
    warm = report.warm(runtime.warmup_min_bars)
    print(f"Darwinex signal cache rebuilt. Warm (>= {runtime.warmup_min_bars} bars): {list(warm)}")
    cold = report.cold(runtime.warmup_min_bars)
    if cold:
        print(f"  WARNING: still cold (will stay flat): {list(cold)}")
    missing = tuple(t.ticker for t in report.per_ticker if t.total_bars == 0)
    if missing:
        print(f"  WARNING: no Darwinex data (scraper not run?) for: {list(missing)}")


def _run_armed(node, minutes: float) -> None:
    stopper: threading.Timer | None = None
    if minutes > 0:
        stopper = threading.Timer(minutes * 60.0, node.stop)
        stopper.daemon = True
        stopper.start()
        print(f"Auto-stop scheduled in {minutes:g} minute(s).")

    def _shutdown(_sig, _frame):
        print("\nSignal received — stopping node...")
        node.stop()

    signal.signal(signal.SIGINT, _shutdown)
    try:
        node.run()
    finally:
        if stopper is not None:
            stopper.cancel()
        node.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vault portfolio live on MT5.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="runtime config JSON")
    parser.add_argument("--broker", default=None, help="override the broker in the config (e.g. ftmo)")
    parser.add_argument("--exec-tier", choices=[t.value for t in ExecTier], default=None,
                        help="override exec tier (default from config; sandbox = local fills)")
    parser.add_argument("--live", action="store_true",
                        help="required to ARM a real-order (demo/live) tier")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-build", action="store_true", help="build the node WITHOUT connecting")
    mode.add_argument("--arm", action="store_true", help="connect to the terminal and run")
    parser.add_argument("--minutes", type=float, default=0.0, help="auto-stop after N minutes (0=run until Ctrl+C)")
    parser.add_argument("--skip-refresh", action="store_true",
                        help="(arm) skip the Darwinex signal-cache refresh (use the existing signal cache)")
    args = parser.parse_args()

    runtime = LiveRuntimeConfig.from_json(args.config)
    if args.broker:
        runtime = dataclasses.replace(runtime, broker=args.broker)
    tier = ExecTier(args.exec_tier) if args.exec_tier else ExecTier(runtime.exec_tier)
    print(f"Runtime: broker={runtime.broker} venue={runtime.venue} vault={runtime.vault_root} "
          f"exec_tier={tier.value} tickers={list(runtime.tickers)}")

    try:
        creds = _load_credentials(runtime.broker, tier, allow_dummy=args.dry_build)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")

    if args.dry_build:
        node = build_node(runtime, creds, connect=False, exec_tier=tier)
        exec_desc = "SANDBOX/NETTING (local fills)" if tier is ExecTier.SANDBOX else f"MT5 REAL orders ({tier.value})"
        print(f"✓ dry-build OK — node assembled (data=MT5 live, exec={exec_desc}), "
              "strategy + heartbeat registered. Not connected.")
        node.dispose()
        return

    _assert_armable(tier, creds, live_flag=args.live)
    if tier is not ExecTier.SANDBOX:
        print(f"⚠️  REAL ORDERS: arming exec_tier={tier.value} on {runtime.broker} "
              f"account {creds.login} ({creds.server}).")
    if not args.skip_refresh:
        print("Refreshing shared Darwinex signal cache (scraped data/mt5_data — no broker fetch)...")
        _refresh_signal_data(runtime)
    print(f"Arming node (exec_tier={tier.value}) — connecting to MT5 terminal + loading instruments...")
    node = build_node(runtime, creds, connect=True, exec_tier=tier)
    _run_armed(node, args.minutes)


if __name__ == "__main__":
    main()
