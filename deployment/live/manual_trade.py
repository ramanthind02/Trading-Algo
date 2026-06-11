"""Manual operator CLI for the vault book on MT5 demo accounts.

Complements the automated rollover runtime (``deployment.live.run_vault_sandbox``):
a one-shot, no-Nautilus way to PREVIEW today's target, ENTER it directly, FLATTEN a
book, snapshot ACCOUNT state, or run a self-cleaning ROUNDTRIP exec test. It reuses
the production forecast (:class:`VaultForecastEngine`) + sizing
(``compute_target_signed_lots`` + ``net_rebalance``) + symbol resolution
(``brokers.resolve``), so sizes match what the runtime would place.

HARD SAFETY: every order path aborts unless the bound account is a **DEMO** account
(``account_info().trade_mode == 0``). Orders are tagged **magic 510** (production) so
the rollover runtime recognises them as ours on reconciliation.

Usage (repo root, ``.venv`` — the nautilus-enabled interpreter)::

    python -m deployment.live.manual_trade preview  [--broker ftmo darwinex]
    python -m deployment.live.manual_trade account  [--broker ftmo darwinex]
    python -m deployment.live.manual_trade enter     --broker ftmo darwinex
    python -m deployment.live.manual_trade flatten    --broker ftmo --magic 510   # or --magic all
    python -m deployment.live.manual_trade roundtrip   --broker ftmo
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

from lib.core.repo_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from lib.core.runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()  # UTF-8 console + .env

import MetaTrader5 as mt5  # noqa: E402

from data_platform.providers.mt5 import brokers  # noqa: E402
from deployment.live.broker_data import bind_signal_cache  # noqa: E402
from deployment.live.forecast_engine import ForecastEngineConfig, VaultForecastEngine  # noqa: E402
from deployment.live.runtime.sizing import net_rebalance  # noqa: E402
from execution.mt5_models import OrderSide  # noqa: E402
from execution.mt5_rebalancer import SizingConfig, compute_target_signed_lots  # noqa: E402

MAGIC = 510
DEVIATION = 30
MIN_REBALANCE_LOTS = 0.01
MIN_REBALANCE_NOTIONAL = 50.0
TEST_MAGIC = 990510
DEFAULT_BROKERS = ("ftmo", "darwinex")
_MODE = {0: "DEMO", 1: "CONTEST", 2: "REAL"}


@contextmanager
def _session(broker: str, *, require_demo: bool):
    """Attach to ``broker``'s terminal; yield ``account_info`` or abort if not demo."""
    path = brokers.terminal_path(broker)
    if not mt5.initialize(path=path):
        raise SystemExit(f"[{broker}] mt5.initialize(path={path}) failed: {mt5.last_error()}")
    try:
        mt5.symbols_get()  # populate the symbol table (needed after a terminal switch)
        ai = mt5.account_info()
        if ai is None:
            raise SystemExit(f"[{broker}] no account_info: {mt5.last_error()}")
        if require_demo and ai.trade_mode != 0:
            raise SystemExit(f"[{broker}] ABORT: account {ai.login} is NOT demo "
                             f"(trade_mode={_MODE.get(ai.trade_mode, ai.trade_mode)}). No orders sent.")
        yield ai
    finally:
        mt5.shutdown()


def _filling(si) -> int:
    if si.filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    if si.filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def _live_quote(sym: str):
    """(symbol_info, bid, ask) once a non-zero quote streams, else (si, 0, 0)."""
    mt5.symbol_select(sym, True)
    for _ in range(40):  # ~4s
        si = mt5.symbol_info(sym)
        t = mt5.symbol_info_tick(sym) if si else None
        if t and t.bid > 0 and t.ask > 0:
            return si, float(t.bid), float(t.ask)
        time.sleep(0.1)
    return mt5.symbol_info(sym), 0.0, 0.0


def compute_targets() -> dict[str, float]:
    """Today's vault target ``position_fraction`` per canonical ticker (no orders)."""
    bind_signal_cache()
    engine = VaultForecastEngine(ForecastEngineConfig(vault_root="vault")).load()
    result = engine.evaluate(as_of=datetime.now(timezone.utc))
    if not result.ready:
        raise SystemExit(f"forecast NOT ready (warmup pending): {result.warmup.not_ready()}")
    print(f"signal as_of={result.as_of}  targets="
          f"{ {k: round(v, 4) for k, v in result.targets.items()} }")
    return dict(result.targets)


def _plan_leg(broker: str, canonical: str, frac: float, equity: float):
    """Return (native, symbol_info, mid, target_lots, current_lots, order) or None."""
    try:
        native = brokers.resolve(broker, canonical)
    except Exception as exc:  # ValueError(UNKNOWN) / KeyError
        print(f"  {canonical:4s} unresolved on {broker} ({type(exc).__name__}); skip")
        return None
    si, bid, ask = _live_quote(native)
    if si is None or bid <= 0:
        print(f"  {canonical:4s} {native:12s} no live quote; skip")
        return None
    mid = (bid + ask) / 2.0
    shim = SimpleNamespace(trade_contract_size=float(si.trade_contract_size),
                           volume_step=float(si.volume_step), volume_min=float(si.volume_min),
                           volume_max=float(si.volume_max))
    sizing = SizingConfig(sizing_basis_usd=equity, lot_size_ceiling=100.0,
                          min_rebalance_lots=MIN_REBALANCE_LOTS,
                          min_rebalance_notional_usd=MIN_REBALANCE_NOTIONAL)
    target = compute_target_signed_lots(position_fraction=frac, price=mid, symbol=shim, sizing=sizing)
    ours = [p for p in (mt5.positions_get(symbol=native) or []) if p.magic == MAGIC]
    current = sum((p.volume if p.type == mt5.POSITION_TYPE_BUY else -p.volume) for p in ours)
    order = net_rebalance(target_signed_lots=(target or 0.0), current_signed_lots=current,
                          symbol=shim, price=mid, min_rebalance_lots=MIN_REBALANCE_LOTS,
                          min_rebalance_notional_usd=MIN_REBALANCE_NOTIONAL)
    return native, si, mid, target, current, order, shim, bid, ask


# ── subcommands ───────────────────────────────────────────────────────────────

def cmd_account(brokers_: tuple[str, ...]) -> None:
    for b in brokers_:
        with _session(b, require_demo=False) as ai:
            pos = mt5.positions_get() or []
            ours = sum(1 for p in pos if p.magic == MAGIC)
            print(f"[{b.upper():8s}] {ai.login} {ai.server} [{_MODE.get(ai.trade_mode, ai.trade_mode)}] "
                  f"equity=${ai.equity:,.2f} balance=${ai.balance:,.2f} "
                  f"margin=${ai.margin:,.2f} free=${ai.margin_free:,.2f} "
                  f"positions={len(pos)} (magic{MAGIC}={ours})")


def cmd_preview(brokers_: tuple[str, ...]) -> None:
    targets = compute_targets()
    for b in brokers_:
        with _session(b, require_demo=False) as ai:
            print(f"\n===== PREVIEW {b.upper()}  {ai.login} equity=${ai.equity:,.2f} =====")
            hdr = f"{'TICK':5s} {'SYMBOL':12s} {'frac':>8s} {'mid':>11s} {'tgt_lots':>9s} {'cur':>6s} {'order':>14s}"
            print(hdr); print("-" * len(hdr))
            for canonical, frac in sorted(targets.items(), key=lambda kv: -abs(kv[1])):
                planned = _plan_leg(b, canonical, frac, float(ai.equity))
                if planned is None:
                    continue
                native, si, mid, target, current, order, *_ = planned
                desc = "no-op" if order is None else f"{order.side.value} {order.volume:g}"
                print(f"{canonical:5s} {native:12s} {frac:+8.4f} {mid:11.2f} "
                      f"{(target if target is not None else float('nan')):+9.2f} {current:6.2f} {desc:>14s}")


def cmd_enter(brokers_: tuple[str, ...]) -> None:
    targets = compute_targets()
    for b in brokers_:
        with _session(b, require_demo=True) as ai:
            print(f"\n===== ENTER {b.upper()}  {ai.login} [{_MODE.get(ai.trade_mode)}] "
                  f"equity=${ai.equity:,.2f} =====")
            placed = 0
            for canonical, frac in sorted(targets.items(), key=lambda kv: -abs(kv[1])):
                planned = _plan_leg(b, canonical, frac, float(ai.equity))
                if planned is None:
                    continue
                native, si, mid, target, current, order, shim, bid, ask = planned
                if order is None:
                    print(f"  {canonical:4s} {native:12s} -> no-op (dead-band/at target)")
                    continue
                is_buy = order.side == OrderSide.BUY
                price = ask if is_buy else bid
                r = mt5.order_send(dict(
                    action=mt5.TRADE_ACTION_DEAL, symbol=native, volume=float(order.volume),
                    type=mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL, price=price,
                    deviation=DEVIATION, magic=MAGIC, comment="vault-manual",
                    type_time=mt5.ORDER_TIME_GTC, type_filling=_filling(si)))
                ok = r.retcode == mt5.TRADE_RETCODE_DONE
                placed += int(ok)
                print(f"  {canonical:4s} {native:12s} {'BUY' if is_buy else 'SELL'} {order.volume:g} "
                      f"@~{price:.2f}  retcode={r.retcode} ({r.comment}) {'OK' if ok else 'FAIL'}")
            print(f"  -> {placed} orders filled")
            _report_book(b)


def cmd_flatten(brokers_: tuple[str, ...], magic_arg: str) -> None:
    for b in brokers_:
        with _session(b, require_demo=True) as ai:
            pos = list(mt5.positions_get() or [])
            if magic_arg != "all":
                pos = [p for p in pos if p.magic == int(magic_arg)]
            print(f"[{b.upper()}] {ai.login}: closing {len(pos)} position(s) (filter={magic_arg})")
            for p in pos:
                mt5.symbol_select(p.symbol, True)
                si = mt5.symbol_info(p.symbol)
                t = mt5.symbol_info_tick(p.symbol)
                ctype = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = t.bid if ctype == mt5.ORDER_TYPE_SELL else t.ask
                r = mt5.order_send(dict(
                    action=mt5.TRADE_ACTION_DEAL, symbol=p.symbol, volume=p.volume, type=ctype,
                    position=p.ticket, price=price, deviation=50, magic=p.magic, comment="flatten",
                    type_time=mt5.ORDER_TIME_GTC, type_filling=_filling(si)))
                print(f"    close {p.symbol} ticket={p.ticket} vol={p.volume}: "
                      f"retcode={r.retcode} ({r.comment})")
            time.sleep(1.0)
            _report_book(b, magic=None if magic_arg == "all" else int(magic_arg))


def cmd_roundtrip(brokers_: tuple[str, ...], symbol: str) -> None:
    for b in brokers_:
        with _session(b, require_demo=True) as ai:
            print(f"[{b.upper()}] {ai.login} [{_MODE.get(ai.trade_mode)}] round-trip {symbol}")
            si, bid, ask = _live_quote(symbol)
            if si is None or ask <= 0:
                print(f"  no quote for {symbol}; skip"); continue
            fill = _filling(si)
            r = mt5.order_send(dict(action=mt5.TRADE_ACTION_DEAL, symbol=symbol, volume=float(si.volume_min),
                                    type=mt5.ORDER_TYPE_BUY, price=ask, deviation=DEVIATION, magic=TEST_MAGIC,
                                    comment="rt-open", type_time=mt5.ORDER_TIME_GTC, type_filling=fill))
            print(f"  OPEN BUY {si.volume_min} {symbol}: retcode={r.retcode} ({r.comment})")
            if r.retcode != mt5.TRADE_RETCODE_DONE:
                continue
            time.sleep(1.0)
            for p in [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == TEST_MAGIC]:
                t = mt5.symbol_info_tick(symbol)
                cr = mt5.order_send(dict(action=mt5.TRADE_ACTION_DEAL, symbol=symbol, volume=p.volume,
                                         type=mt5.ORDER_TYPE_SELL, position=p.ticket, price=t.bid,
                                         deviation=DEVIATION, magic=TEST_MAGIC, comment="rt-close",
                                         type_time=mt5.ORDER_TIME_GTC, type_filling=fill))
                print(f"  CLOSE ticket={p.ticket}: retcode={cr.retcode} ({cr.comment})")
            left = [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == TEST_MAGIC]
            print(f"  residual TEST_MAGIC: {len(left)} -> {'FLAT [OK]' if not left else 'NOT FLAT [FAIL]'}")


def _report_book(broker: str, magic: int | None = MAGIC) -> None:
    pos = [p for p in (mt5.positions_get() or []) if magic is None or p.magic == magic]
    ai = mt5.account_info()
    gross = 0.0
    for p in pos:
        side = "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT"
        si = mt5.symbol_info(p.symbol)
        notion = p.volume * p.price_current * (si.trade_contract_size if si else 1)
        gross += abs(notion)
        print(f"     {p.symbol:12s} {side} {p.volume:g} @ {p.price_open:.2f} "
              f"P&L=${p.profit:+.2f} (~${notion:,.0f})")
    label = "all" if magic is None else f"magic{magic}"
    print(f"     {len(pos)} {label} positions, gross ~${gross:,.0f} | equity=${ai.equity:,.2f} "
          f"free=${ai.margin_free:,.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Manual operator CLI for the vault book on MT5 demo accounts.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("preview", "account"):
        s = sub.add_parser(name)
        s.add_argument("--broker", nargs="+", default=list(DEFAULT_BROKERS))
    for name in ("enter",):
        s = sub.add_parser(name)
        s.add_argument("--broker", nargs="+", required=True)
    s = sub.add_parser("flatten")
    s.add_argument("--broker", nargs="+", required=True)
    s.add_argument("--magic", default=str(MAGIC), help="magic to close, or 'all'")
    s = sub.add_parser("roundtrip")
    s.add_argument("--broker", nargs="+", required=True)
    s.add_argument("--symbol", default="EURUSD")
    args = p.parse_args()

    brokers_ = tuple(args.broker)
    if args.cmd == "account":
        cmd_account(brokers_)
    elif args.cmd == "preview":
        cmd_preview(brokers_)
    elif args.cmd == "enter":
        cmd_enter(brokers_)
    elif args.cmd == "flatten":
        cmd_flatten(brokers_, args.magic)
    elif args.cmd == "roundtrip":
        cmd_roundtrip(brokers_, args.symbol)


if __name__ == "__main__":
    main()
