r"""Daily guard: FAIL LOUD if the MT5 signal feed has gone stale.

The live vault signal is built from the daily MT5 scrape (``data/mt5_data``). If that
scrape stops — a broken Task Scheduler entry, a downed terminal, a moved path — the feed
silently freezes and the nodes keep trading an old forecast (exactly what happened: the
feed was ~2 trading days stale and nobody noticed). This guard checks the freshest stored
bar for the vault signal symbols and, when it is older than ``--max-age-days``, it:

  * writes a status file (``logs/data_freshness.json``) with a ``stale`` flag,
  * logs an ERROR + prints a clear STALE verdict,
  * sends a Telegram alert IF one is configured (logs-only otherwise),
  * exits non-zero (so a wrapping ``.bat`` / Task Scheduler marks the run failed).

Run it BOTH right after the scrape (catches a scrape that ran but didn't advance) AND as
its own daily task (catches a scrape task that never fired at all) — two independent
tripwires.

Note on the clock: this box's system clock is unreliable, so the age is coarse by design
(default threshold 3 days) — enough to catch a multi-day freeze without false alarms from
a few hours of host-clock drift or the normal 1-day scrape-vs-entry lag.

Usage::

    .\.venv\Scripts\python.exe -m deployment.ops.check_data_freshness
    .\.venv\Scripts\python.exe -m deployment.ops.check_data_freshness --max-age-days 3 --tickers ES NQ GC CL SI
    .\.venv\Scripts\python.exe -m deployment.ops.check_data_freshness --parity-check
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = next((p for p in _HERE.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
                  _HERE.parents[2])
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.core.runtime_bootstrap import bootstrap_runtime  # noqa: E402

bootstrap_runtime(_REPO_ROOT)

import pyarrow.parquet as pq  # noqa: E402

from data_platform.providers.mt5 import brokers  # noqa: E402
from lib.core.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

MT5_DATA_DIR = _REPO_ROOT / "data" / "mt5_data"
STATUS_PATH = _REPO_ROOT / "logs" / "data_freshness.json"
DEFAULT_TICKERS = ("ES", "NQ", "GC", "CL", "SI")
SIGNAL_BROKER = "darwinex"  # the feed the live vault signal is built from


# ---------------------------------------------------------------------------
# Legacy file-based path (parquet footer reads)
# ---------------------------------------------------------------------------

def _freshness_from_files(symbol: str) -> datetime | None:
    """Latest M1 bar timestamp stored for ``symbol`` (UTC-aware), or None.

    Reads the parquet footer of the last year-partition file — no full
    table scan, just metadata.  This is the fallback path when the registry
    DB is unavailable.
    """
    files = sorted(glob.glob(str(MT5_DATA_DIR / symbol / "bars_M1" / "year=*" / "part.parquet")))
    if not files:
        return None
    try:
        col = pq.read_table(files[-1], columns=["time"]).column("time").to_pylist()
        if not col:
            return None
        ts = col[-1]
        ts = ts if isinstance(ts, datetime) else datetime.fromtimestamp(0, tz=timezone.utc)
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("freshness: could not read %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Registry-backed path
# ---------------------------------------------------------------------------

def _query_freshness_from_registry(
    tickers: list[str],
    broker: str = SIGNAL_BROKER,
    *,
    db_path: Path | None = None,
) -> dict[str, datetime | None]:
    """Query registry DB for freshest mt5_m1 bar per vault symbol.

    Returns ``{'{canonical}->{broker_sym}': datetime | None}`` — same key
    format as the legacy per_symbol dict so callers can use either path
    transparently.

    Raises ``FileNotFoundError`` (DB missing), ``RegistryVersionError``
    (schema mismatch), or any other registry error — the caller is expected
    to catch and fall back to the file path.
    """
    from data_platform.registry import db as _reg_db, reader as _reg_reader

    conn = _reg_db.connect_readonly(db_path)
    try:
        rows = _reg_reader.freshness(conn, "mt5_m1")
    finally:
        conn.close()

    # Aggregate MAX coverage_end per symbol across all year partitions.
    # key_json for mt5_m1 is {"symbol": "SP500", "year": "2024"} so we need
    # an extra GROUP-BY at the Python level.
    symbol_freshest: dict[str, str] = {}
    for row in rows:
        key = json.loads(row["key_json"])
        sym = key.get("symbol")
        end = row["coverage_end"]   # date string e.g. "2026-06-08" or None
        if sym and end:
            if sym not in symbol_freshest or end > symbol_freshest[sym]:
                symbol_freshest[sym] = end

    result: dict[str, datetime | None] = {}
    for canonical in tickers:
        try:
            sym = brokers.resolve(broker, canonical)
        except Exception:  # noqa: BLE001
            continue
        cov_end = symbol_freshest.get(sym)
        dt: datetime | None
        if cov_end:
            # Convert date string → midnight UTC datetime for age arithmetic.
            dt = datetime.strptime(cov_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            dt = None
        result[f"{canonical}->{sym}"] = dt

    return result


# ---------------------------------------------------------------------------
# Telegram alert
# ---------------------------------------------------------------------------

def _telegram_alert(text: str) -> bool:
    """Best-effort Telegram alert; returns False (logs-only) when unconfigured."""
    try:
        from lib.core.notify import TelegramNotifier
        notifier = TelegramNotifier()
        if not getattr(notifier, "token", None) or not getattr(notifier, "chat_id", None):
            return False
        return bool(notifier.send_message(text))
    except Exception as exc:  # noqa: BLE001
        logger.warning("freshness: telegram alert failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Core logic (extracted for testability)
# ---------------------------------------------------------------------------

def _compute_freshness(
    tickers: list[str],
    now: datetime,
    max_age_days: float,
    per_symbol: dict[str, datetime | None],
) -> dict:
    """Build the status dict from pre-resolved per_symbol data."""
    freshest: datetime | None = None
    per_symbol_iso: dict[str, str | None] = {}
    for key, latest in per_symbol.items():
        per_symbol_iso[key] = latest.isoformat() if latest else None
        if latest and (freshest is None or latest > freshest):
            freshest = latest

    age_days = None if freshest is None else round(
        (now - freshest).total_seconds() / 86400.0, 2
    )
    stale = freshest is None or age_days > max_age_days
    return {
        "checked_at": now.isoformat(),
        "freshest_bar": freshest.isoformat() if freshest else None,
        "age_days": age_days,
        "max_age_days": max_age_days,
        "stale": stale,
        "per_symbol": per_symbol_iso,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Alert if the MT5 signal feed is stale.")
    ap.add_argument("--max-age-days", type=float, default=3.0,
                    help="flag stale when the freshest vault-symbol bar is older than this (default 3)")
    ap.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    ap.add_argument(
        "--parity-check",
        action="store_true",
        help=(
            "Compare registry vs legacy-file paths and print both results; "
            "writes to logs/data_freshness_compare.json instead of the canonical "
            "logs/data_freshness.json so the production file is not overwritten "
            "until parity is confirmed."
        ),
    )
    args = ap.parse_args()

    now = datetime.now(timezone.utc)

    if args.parity_check:
        return _run_parity_check(args.tickers, now, args.max_age_days)

    # ── Normal run: registry first, file fallback ─────────────────────────────
    source = "registry"
    per_symbol: dict[str, datetime | None] = {}

    try:
        per_symbol = _query_freshness_from_registry(args.tickers, SIGNAL_BROKER)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "freshness: registry unavailable (%s) — falling back to parquet file reads",
            exc,
        )
        source = "files"
        for canonical in args.tickers:
            try:
                sym = brokers.resolve(SIGNAL_BROKER, canonical)
            except Exception:  # noqa: BLE001
                continue
            per_symbol[f"{canonical}->{sym}"] = _freshness_from_files(sym)

    status = _compute_freshness(args.tickers, now, args.max_age_days, per_symbol)
    status["source"] = source

    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")

    stale = status["stale"]
    freshest_str = status["freshest_bar"]
    age_days = status["age_days"]

    if stale:
        msg = (f"⚠️ MT5 SIGNAL FEED STALE: freshest vault bar = {freshest_str} "
               f"({age_days}d old > {args.max_age_days}d). The daily scrape is not advancing "
               f"the feed — the live nodes are trading an OLD forecast. Check the MT5DataScrape "
               f"task + logs/mt5_scrape.log.")
        logger.error(msg)
        print(f"STALE: {msg}")
        sent = _telegram_alert(msg)
        print(f"(telegram alert {'sent' if sent else 'NOT configured — logs/status only'})")
        return 1

    print(f"OK: signal feed fresh — freshest vault bar {freshest_str} ({age_days}d old, "
          f"<= {args.max_age_days}d) [source={source}]. status -> {STATUS_PATH}")
    return 0


def _run_parity_check(
    tickers: list[str],
    now: datetime,
    max_age_days: float,
) -> int:
    """Run both paths and compare; print results; write to compare file."""
    compare_path = _REPO_ROOT / "logs" / "data_freshness_compare.json"

    # Registry path
    reg_per_symbol: dict[str, datetime | None] = {}
    reg_error: str | None = None
    try:
        reg_per_symbol = _query_freshness_from_registry(tickers, SIGNAL_BROKER)
    except Exception as exc:  # noqa: BLE001
        reg_error = str(exc)
        logger.warning("parity: registry path failed: %s", exc)

    # File path
    files_per_symbol: dict[str, datetime | None] = {}
    for canonical in tickers:
        try:
            sym = brokers.resolve(SIGNAL_BROKER, canonical)
        except Exception:  # noqa: BLE001
            continue
        files_per_symbol[f"{canonical}->{sym}"] = _freshness_from_files(sym)

    reg_status = (
        _compute_freshness(tickers, now, max_age_days, reg_per_symbol)
        if not reg_error
        else {"error": reg_error}
    )
    file_status = _compute_freshness(tickers, now, max_age_days, files_per_symbol)

    compare = {
        "checked_at": now.isoformat(),
        "registry": reg_status,
        "files": file_status,
    }

    compare_path.parent.mkdir(parents=True, exist_ok=True)
    compare_path.write_text(json.dumps(compare, indent=2), encoding="utf-8")

    print(f"\n=== Parity check written to {compare_path} ===\n")
    print(f"Registry path:  freshest_bar={reg_status.get('freshest_bar')!r}  "
          f"stale={reg_status.get('stale')!r}  age_days={reg_status.get('age_days')!r}")
    print(f"Files    path:  freshest_bar={file_status.get('freshest_bar')!r}  "
          f"stale={file_status.get('stale')!r}  age_days={file_status.get('age_days')!r}")

    # Agreement check: same staleness verdict + freshest_bar within same day
    reg_fb = reg_status.get("freshest_bar")
    file_fb = file_status.get("freshest_bar")
    verdicts_match = reg_status.get("stale") == file_status.get("stale")
    dates_match = (
        reg_fb is not None
        and file_fb is not None
        and reg_fb[:10] == file_fb[:10]  # YYYY-MM-DD prefix
    )
    if reg_error:
        print(f"\nREGISTRY PATH FAILED: {reg_error}")
        print("Run: python -m data_platform.registry rebuild --domains manifest  "
              "to populate the registry, then re-run --parity-check.")
        return 1
    if verdicts_match and dates_match:
        print("\nPARITY OK — both paths agree on staleness verdict and freshest_bar date.")
        return 0

    print(
        "\nPARITY MISMATCH — registry and file paths disagree. "
        "Registry may need a manifest rebuild."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
