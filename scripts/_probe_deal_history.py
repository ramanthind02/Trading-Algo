"""
M0.7 probe: how far back does history_deals_get() reach per broker?

Run ONLY when:
  - No live node (run_vault_sandbox.py) is active — MT5 IPC is single-process.
  - At least one terminal64.exe is running and logged in.

Usage:
    .\.venv\Scripts\python.exe scripts\_probe_deal_history.py

Prints a summary table: broker | oldest_deal_date | total_deals | login
"""
from __future__ import annotations

import datetime
import time
from pathlib import Path

import MetaTrader5 as mt5

import data_platform.providers.mt5.brokers as brokers

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BROKERS_TO_PROBE: list[str] = ["ftmo", "darwinex", "fundednext"]

# Date ladder: probe from increasingly early dates.
# history_deals_get(from_date, to_date) returns all deals in [from_date, to_date].
DATE_LADDER: list[datetime.datetime] = [
    datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
    datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
    datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc),
    datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc),
    datetime.datetime(2018, 1, 1, tzinfo=datetime.timezone.utc),
]

SHUTDOWN_PAUSE_S: int = 5  # seconds to wait between brokers (MT5 IPC teardown)


# ---------------------------------------------------------------------------
# Probe one broker
# ---------------------------------------------------------------------------

def _probe_broker(broker_name: str, now: datetime.datetime) -> dict:
    result = {
        "broker": broker_name,
        "login": None,
        "oldest_deal_date": None,
        "total_deals": 0,
        "error": None,
    }

    # Check terminal path exists.
    path_str = brokers.terminal_path(broker_name)
    if path_str is None:
        result["error"] = "no terminal_path in config"
        return result

    path = Path(path_str)
    if not path.exists():
        result["error"] = f"terminal not found: {path_str}"
        return result

    print(f"\n{'='*60}")
    print(f"  Probing broker: {broker_name}")
    print(f"  Terminal: {path_str}")

    # Initialize (attach mode — path only, no credentials).
    ok = mt5.initialize(path=str(path))
    if not ok:
        code, msg = mt5.last_error()
        result["error"] = f"mt5.initialize() failed: error {code}: {msg}"
        print(f"  ERROR: {result['error']}")
        return result

    # Account info.
    info = mt5.account_info()
    if info is None:
        code, msg = mt5.last_error()
        result["error"] = f"mt5.account_info() returned None: error {code}: {msg}"
        print(f"  ERROR: {result['error']}")
        mt5.shutdown()
        return result

    result["login"] = info.login
    print(f"  Login: {info.login}  Server: {info.server}  Balance: {info.balance:.2f} {info.currency}")

    # Walk the date ladder to find the oldest reachable deals.
    # We start from the most recent cutoff that still has deals,
    # then try earlier and earlier.
    all_deals: list = []
    oldest_from: datetime.datetime | None = None

    # First: quick check from 2024-01-01 to confirm any deals exist at all.
    for from_dt in DATE_LADDER:
        print(f"  Trying from {from_dt.date()} ...", end=" ", flush=True)
        deals = mt5.history_deals_get(from_dt, now)

        if deals is None:
            code, msg = mt5.last_error()
            # Error -2 = "history not loaded" — no deals in this range (normal).
            # Other codes are real errors.
            if code not in (-2, 0):
                print(f"ERROR (code={code}: {msg})")
            else:
                print("0 deals")
            # No deals from this date — stop trying to go further back.
            # (If a newer date already had deals, the oldest anchor is set.)
            break

        n = len(deals)
        print(f"{n} deals")

        if n > 0:
            # Keep the best (earliest) anchor.
            oldest_from = from_dt
            all_deals = list(deals)
            # Keep going back.
        else:
            # Empty result from this date — no point going further back.
            break

    if all_deals:
        # Find the actual oldest deal date in the returned set.
        min_ts = min(d.time for d in all_deals)
        result["oldest_deal_date"] = datetime.datetime.fromtimestamp(
            min_ts, tz=datetime.timezone.utc
        )
        result["total_deals"] = len(all_deals)

        print(f"  -> Oldest deal date: {result['oldest_deal_date'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  -> Total deals (from {oldest_from.date()} to now): {result['total_deals']}")

        # Print a few sample deals (most recent).
        print(f"  -> Most recent 3 deals:")
        recent = sorted(all_deals, key=lambda d: d.time, reverse=True)[:3]
        for d in recent:
            ts = datetime.datetime.fromtimestamp(d.time, tz=datetime.timezone.utc)
            print(f"       ticket={d.ticket}  {ts.strftime('%Y-%m-%d %H:%M')}  "
                  f"{d.symbol:>12}  vol={d.volume:.2f}  price={d.price:.5f}  "
                  f"profit={d.profit:.2f}")
    else:
        print(f"  -> No deals found in any range.")

    mt5.shutdown()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    print(f"\nMT5 Deal History Probe — {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"NOTE: host clock may be unreliable (~7h fast observed). Using UTC probe dates.")
    print(f"Brokers to probe: {', '.join(BROKERS_TO_PROBE)}")

    results: list[dict] = []

    for i, broker_name in enumerate(BROKERS_TO_PROBE):
        result = _probe_broker(broker_name, now)
        results.append(result)

        # Pause between brokers to allow MT5 IPC to release cleanly.
        if i < len(BROKERS_TO_PROBE) - 1:
            print(f"\n  Waiting {SHUTDOWN_PAUSE_S}s before next broker...")
            time.sleep(SHUTDOWN_PAUSE_S)

    # Summary table.
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Broker':<14} {'Login':<12} {'Oldest Deal':<22} {'Total Deals':<12} {'Notes'}")
    print(f"{'-'*14} {'-'*12} {'-'*22} {'-'*12} {'-'*20}")

    for r in results:
        broker = r["broker"]
        login = str(r["login"]) if r["login"] else "N/A"
        oldest = r["oldest_deal_date"].strftime("%Y-%m-%d") if r["oldest_deal_date"] else "N/A"
        total = str(r["total_deals"]) if r["total_deals"] else "0"
        notes = r["error"] or ""
        print(f"{broker:<14} {login:<12} {oldest:<22} {total:<12} {notes}")

    print()

    # Assessment.
    any_deals = any(r["total_deals"] > 0 for r in results)
    if any_deals:
        print("Assessment:")
        print("  Our live positions entered ~2026-06-08.")
        for r in results:
            if r["total_deals"] > 0 and r["oldest_deal_date"]:
                cutoff = datetime.datetime(2026, 6, 8, tzinfo=datetime.timezone.utc)
                reaches_live = r["oldest_deal_date"] <= cutoff
                label = "COVERS live positions" if reaches_live else "does NOT reach live positions"
                print(f"  {r['broker']}: oldest={r['oldest_deal_date'].date()} -> {label}")
    else:
        print("  No deals found on any broker — terminals may not be logged in.")


if __name__ == "__main__":
    main()
