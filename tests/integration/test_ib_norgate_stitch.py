r"""Live integration test: IB CONTFUT -> Norgate stitching (Norgate->IB handover).

Connects to a running TWS / IB Gateway, fetches CONTFUT daily bars for a set of
futures, splices them onto the frozen Norgate anchor (data/ohlc_data) via the
SourcePriorityReconciler in handover mode (norgate_active=False), and asserts the
spliced series has NO price gap at the junction and no abnormal jumps.

Requires a live IB connection — skipped automatically if TWS is not reachable.

Run directly (verbose, prints the splice diagnostics):
    .\.venv\Scripts\python.exe -m tests.integration.test_ib_norgate_stitch --port 7497

Or via pytest (auto-skips if IB unreachable):
    .\.venv\Scripts\python.exe -m pytest tests/integration/test_ib_norgate_stitch.py -v -s
"""
from __future__ import annotations

import argparse
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data_platform.core import load_catalog, source_symbol
from data_platform.core.reconciler import SourcePriorityReconciler
from data_platform.core.source_priority import SourcePriorityConfig
from data_platform.loaders import load_data
from lib.core.enums import Ticker, TimeFrame

DEFAULT_PORT = 7497
DEFAULT_TICKERS = ["ES", "NQ", "GC"]
# Max acceptable day-over-day return at the splice boundary (post-ratio). A clean
# splice makes the first IB close == anchor close, so boundary return ~ 0; the
# following sessions should look like normal daily moves, not a roll-gap jump.
MAX_BOUNDARY_RETURN = 0.06  # 6% — generous; a broken splice shows 10-40%+


# ── IB client (minimal, reuses the demo client's proven callback shape) ─────

def _make_ib_client(host: str, port: int, client_id: int):
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract

    class _Client(EClient, EWrapper):
        def __init__(self) -> None:
            EClient.__init__(self, self)
            self._next_id = 0
            self.connected_ok = False
            self.bars: list = []
            self.done: dict[int, bool] = {}
            self.errors: dict[int, str] = {}

        def nextValidId(self, orderId: int) -> None:
            self._next_id = orderId
            self.connected_ok = True

        def next_id(self) -> int:
            self._next_id += 1
            return self._next_id

        def error(self, *args) -> None:
            # Tolerant of both old/new ibapi error signatures.
            code = None
            req = args[0] if args else -1
            for a in args[1:]:
                if isinstance(a, int):
                    code = a
                    break
            if code in (2104, 2106, 2158, 2176, 366):
                return
            if isinstance(req, int) and req > 0 and not self.done.get(req, False):
                self.errors[req] = f"code={code}"
                self.done[req] = True

        def historicalData(self, reqId: int, bar) -> None:
            self.bars.append(bar)

        def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
            self.done[reqId] = True

        def fetch_contfut_daily(self, symbol: str, exchange: str, duration: str = "1 Y") -> pd.DataFrame:
            c = Contract()
            c.symbol = symbol
            c.secType = "CONTFUT"
            c.exchange = exchange
            c.currency = "USD"
            req = self.next_id()
            self.bars = []
            self.done[req] = False
            self.reqHistoricalData(req, c, "", duration, "1 day", "TRADES", 0, 1, False, [])
            t0 = time.time()
            while not self.done.get(req, False):
                if time.time() - t0 > 30:
                    break
                time.sleep(0.1)
            rows = [
                {"datetime": pd.to_datetime(b.date.split()[0]),
                 "open": float(b.open), "high": float(b.high),
                 "low": float(b.low), "close": float(b.close),
                 "volume": int(float(b.volume)) if float(b.volume) >= 0 else 0}
                for b in self.bars
            ]
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows).set_index("datetime").sort_index()
            return df

    client = _Client()
    client.connect(host, port, client_id)
    t = threading.Thread(target=client.run, daemon=True)
    t.start()
    t0 = time.time()
    while not client.connected_ok and time.time() - t0 < 8:
        time.sleep(0.1)
    return client if client.connected_ok else None


def _ib_available(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


# ── the stitch check ────────────────────────────────────────────────────────

def _stitch_and_check(ticker: str, ib_df: pd.DataFrame) -> dict:
    """Splice IB CONTFUT onto the Norgate anchor; return diagnostics."""
    catalog = load_catalog()
    instrument_id = next(
        (str(i.id) for i in catalog.all()
         if i.info.get("source_symbols", {}).get("ib_contfut") == ticker
         and i.instrument_class.name == "FUTURE"),
        None,
    )
    assert instrument_id is not None, f"{ticker} not in catalog"

    # Norgate anchor (frozen), indexed by datetime
    anchor = load_data(Ticker[ticker], TimeFrame.D).reset_index()
    anchor = anchor.set_index(pd.to_datetime(anchor["datetime"]))[["open", "high", "low", "close", "volume"]]
    anchor_end = anchor.index.max()
    anchor_close = float(anchor.loc[anchor_end, "close"])

    # Keep only IB sessions strictly after the anchor end (the handover region)
    ib_after = ib_df.loc[ib_df.index > anchor_end]
    if ib_after.empty:
        return {"ticker": ticker, "skipped": "no IB bars after anchor end", "anchor_end": anchor_end}

    cfg = SourcePriorityConfig.default()
    cfg.norgate_active = False  # handover: futures daily -> IB
    rec = SourcePriorityReconciler(cfg, catalog)
    result = rec.reconcile_daily_batch(instrument_id, anchor, {"ib": ib_after}, resolution="D")

    merged = result.merged
    # Boundary return: first appended close vs anchor close (should be ~0 after ratio)
    appended = merged.loc[merged.index > anchor_end].sort_index()
    first_ib_close = float(appended["close"].iloc[0])
    boundary_return = abs(first_ib_close - anchor_close) / anchor_close

    # Max single-day return across the whole appended block (catch residual jumps)
    closes = appended["close"].to_numpy(dtype=float)
    block_returns = np.abs(np.diff(closes) / closes[:-1]) if len(closes) > 1 else np.array([0.0])
    max_block_return = float(block_returns.max()) if block_returns.size else 0.0

    return {
        "ticker": ticker,
        "instrument_id": instrument_id,
        "active_source": result.active_source,
        "ratio_applied": result.provenance[0].ratio_applied,
        "ratio": result.provenance[0].ratio_value,
        "anchor_end": anchor_end.date(),
        "anchor_close": anchor_close,
        "first_ib_close_pre_splice": float(ib_after["close"].iloc[0]),
        "first_ib_close_post_splice": first_ib_close,
        "boundary_return": boundary_return,
        "max_block_return": max_block_return,
        "ib_rows_appended": len(appended),
        "ib_raw_rows": len(ib_df),
    }


# ── pytest entry ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _ib_available(), reason="IB/TWS not reachable on 127.0.0.1:7497")
@pytest.mark.parametrize("ticker", DEFAULT_TICKERS)
def test_ib_norgate_stitch_is_gap_free(ticker: str) -> None:
    client = _make_ib_client("127.0.0.1", DEFAULT_PORT, client_id=77)
    assert client is not None, "failed to connect to IB"
    try:
        ex = source_symbol(load_catalog(), _resolve_id(ticker), "ib_exchange") or "CME"
        ib_df = client.fetch_contfut_daily(ticker, ex, duration="1 Y")
        assert not ib_df.empty, f"no IB bars for {ticker}"
        diag = _stitch_and_check(ticker, ib_df)
        print(f"\n{ticker} stitch diagnostics: {diag}")
        if "skipped" in diag:
            pytest.skip(diag["skipped"])
        assert diag["ratio_applied"] is True
        assert diag["boundary_return"] == pytest.approx(0.0, abs=1e-6), (
            f"{ticker}: price GAP at splice — boundary return {diag['boundary_return']:.4%}")
        assert diag["max_block_return"] < MAX_BOUNDARY_RETURN, (
            f"{ticker}: abnormal jump in spliced block ({diag['max_block_return']:.4%})")
    finally:
        client.disconnect()


def _resolve_id(ticker: str) -> str:
    cat = load_catalog()
    return next(
        str(i.id) for i in cat.all()
        if i.info.get("source_symbols", {}).get("ib_contfut") == ticker
        and i.instrument_class.name == "FUTURE"
    )


# ── standalone runner ────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--duration", default="1 Y")
    args = ap.parse_args()

    if not _ib_available(port=args.port):
        print(f"IB not reachable on 127.0.0.1:{args.port} — start TWS/Gateway and enable API.")
        return
    client = _make_ib_client("127.0.0.1", args.port, client_id=77)
    if client is None:
        print("Failed to connect to IB.")
        return
    try:
        cat = load_catalog()
        print(f"{'ticker':<6} {'anchor_end':<12} {'anchor_close':>12} {'ib_pre':>12} "
              f"{'ib_post':>12} {'ratio':>8} {'boundary':>10} {'max_jump':>9} {'rows':>5}")
        for ticker in args.tickers:
            ex = source_symbol(cat, _resolve_id(ticker), "ib_exchange") or "CME"
            ib_df = client.fetch_contfut_daily(ticker, ex, duration=args.duration)
            if ib_df.empty:
                print(f"{ticker:<6} no IB bars returned")
                continue
            d = _stitch_and_check(ticker, ib_df)
            if "skipped" in d:
                print(f"{ticker:<6} skipped: {d['skipped']}")
                continue
            status = "OK" if (d["boundary_return"] < 1e-6 and d["max_block_return"] < MAX_BOUNDARY_RETURN) else "FAIL"
            print(f"{ticker:<6} {str(d['anchor_end']):<12} {d['anchor_close']:>12.2f} "
                  f"{d['first_ib_close_pre_splice']:>12.2f} {d['first_ib_close_post_splice']:>12.2f} "
                  f"{(d['ratio'] or 0):>8.4f} {d['boundary_return']:>10.2e} "
                  f"{d['max_block_return']:>8.2%} {d['ib_rows_appended']:>5}  [{status}]")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
