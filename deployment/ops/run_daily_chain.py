"""Daily data-platform maintenance chain.

Runs steps a-f in order. Steps b-e are subprocesses so a crash in one cannot
kill the chain; every step's outcome is recorded via the registry job_run idiom
(resilience-wrapped so a missing/locked DB never aborts the chain).

TERMINAL-CONTENTION WARNING
----------------------------
!! The MT5 scraper (step b) attaches to the Darwinex terminal identified by
!! MT5_PATH in .env, which resolves to the SAME terminal64.exe path as
!! configs/mt5_brokers.yaml brokers.darwinex.terminal_path:
!!
!!   C:/Program Files/Darwinex MetaTrader 5/terminal64.exe
!!
!! Plan §8.3 assumes this terminal is DEDICATED to the scraper. The operator
!! MUST NOT arm the Darwinex live node simultaneously with this chain until the
!! Darwinex scraper terminal and the Darwinex live-node terminal are split into
!! two separate MT5 installations.

Steps
-----
  clock_drift     a) Compare host UTC epoch vs broker EET epoch from a fresh
                     EURUSD tick; warn via Telegram when drift from expected
                     EET window exceeds 30 min. Skip silently if terminal
                     is unreachable.
  scrape          b) python -m data_platform.providers.mt5.scraper
  freshness       c) python -m deployment.ops.check_data_freshness
  signal_refresh  d) Populate shared Darwinex signal cache from scraped
                     data/mt5_data (no MT5 connection required).
  catalog_append  e) Append only bars newer than each symbol's current catalog
                     coverage end to the persistent Nautilus ParquetDataCatalog,
                     then consolidate each appended identifier by 1-day period.
                     Skipped gracefully (exit 0) when the catalog is empty/absent
                     or no symbol has been materialized yet.
                     Repair primitive: delete_data_range(Bar, identifier, start,
                     end) + re-run ``python -m data_platform.nautilus.materialize
                     --symbols SYM --from YYYY`` for the affected symbol/span.
  registry_ingest e1) python -m data_platform.registry ingest live
  registry_backup e2) python -m data_platform.registry backup
  (summary)       f) Print exit-code table; Telegram alert on any failed step.

CLI
---
  python -m deployment.ops.run_daily_chain
  python -m deployment.ops.run_daily_chain --skip clock_drift,freshness
  python -m deployment.ops.run_daily_chain --only scrape
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time as _time_module
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _HERE.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
    _HERE.parents[2],
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.core.runtime_bootstrap import bootstrap_runtime  # noqa: E402

bootstrap_runtime(_REPO_ROOT)

from lib.core.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# ── venv resolution ───────────────────────────────────────────────────────────


def _venv_python() -> str:
    """Resolve the venv python interpreter relative to the repo root."""
    for candidate in (
        ".venv/Scripts/python.exe",
        "venv/Scripts/python.exe",
        ".venv/bin/python",
        "venv/bin/python",
    ):
        p = _REPO_ROOT / candidate
        if p.exists():
            return str(p)
    return sys.executable  # fallback: current interpreter


_VENV_PYTHON = _venv_python()

# ── Step names + result ───────────────────────────────────────────────────────

ALL_STEPS = (
    "clock_drift",
    "scrape",
    "freshness",
    "signal_refresh",
    "catalog_append",
    "registry_ingest",
    "registry_backup",
)


class StepResult(NamedTuple):
    name: str
    exit_code: int   # 0=ok  >0=failed  -1=skipped  -2=exception
    duration_s: float
    detail: str = ""


# ── Registry job_run recorder (resilience-wrapped) ───────────────────────────


def _record_start(job_name: str, args: dict | None = None) -> tuple[object, int | None]:
    """Open a registry connection and insert a job_runs row.

    Returns (conn, run_id). Both are None when the registry is unavailable;
    callers must tolerate that gracefully.
    """
    try:
        from data_platform.registry import db as _reg_db, writer as _reg_writer

        conn = _reg_db.connect()
        with _reg_db.transaction(conn):
            run_id = _reg_writer.record_job_run(
                conn,
                _reg_writer.JobRun(
                    job_name=job_name,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    args_json=json.dumps(args or {}),
                ),
            )
        return conn, run_id
    except Exception as exc:  # noqa: BLE001
        logger.debug("registry unavailable for %s: %s", job_name, exc)
        return None, None


def _record_finish(
    conn: object,
    run_id: int | None,
    exit_code: int,
    detail: str,
) -> None:
    if conn is None or run_id is None:
        return
    try:
        from data_platform.registry import db as _reg_db, writer as _reg_writer

        with _reg_db.transaction(conn):
            _reg_writer.finish_job_run(
                conn,
                run_id,
                exit_code=exit_code,
                rows_written=0,
                coverage_json=None,
                error_text=detail if exit_code != 0 else None,
            )
        conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("registry finish failed for run_id=%s: %s", run_id, exc)


# ── Telegram alert (resilience-wrapped) ──────────────────────────────────────


def _send_alert(text: str) -> None:
    """Best-effort Telegram alert; logs-only when CFD-prop channel is unconfigured."""
    try:
        from lib.core.notify import TelegramNotifier

        TelegramNotifier.for_cfd_prop().send_message(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("alert send failed: %s", exc)


# ── Subprocess step runner ────────────────────────────────────────────────────


def _run_step_subprocess(
    step_name: str,
    cmd: list[str],
    *,
    env: dict | None = None,
) -> StepResult:
    """Run a step as a subprocess; the chain continues regardless of exit code."""
    logger.info("STEP %s: %s", step_name, " ".join(str(c) for c in cmd))
    t0 = datetime.now(timezone.utc)
    conn, run_id = _record_start(step_name, {"cmd": [str(c) for c in cmd]})
    try:
        r = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(_REPO_ROOT), **(env or {})},
        )
        code = r.returncode
        detail = "" if code == 0 else f"exit {code}"
    except Exception as exc:  # noqa: BLE001
        code = -2
        detail = str(exc)
        logger.error("STEP %s exception: %s", step_name, exc)
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    _record_finish(conn, run_id, code, detail)
    tag = "OK" if code == 0 else f"FAILED(exit={code})"
    logger.info("STEP %s %s in %.1fs", step_name, tag, elapsed)
    return StepResult(step_name, code, elapsed, detail)


# ── Step a: clock drift ───────────────────────────────────────────────────────


def step_clock_drift() -> StepResult:
    """Compare host UTC epoch vs broker EET epoch (fresh EURUSD tick).

    The Darwinex terminal stores ``tick.time`` as broker wall-clock (EET/EEST)
    in Unix-epoch format. On a correctly-set machine the raw difference
    ``host_utc_epoch - tick.time`` sits in [-10800, -7200] (host is 2-3 h
    behind the broker EET clock). The observed fault was the host clock ~7 h
    FAST, putting this difference near +18 000 s. We flag whenever the
    difference leaves the ±30-min window around the expected EET range.
    """
    t0 = datetime.now(timezone.utc)
    conn, run_id = _record_start("clock_drift")
    try:
        import MetaTrader5 as mt5  # type: ignore[import-untyped]

        from data_platform.providers.mt5 import brokers as _brokers
        from deployment.live.runtime.rollover_market import broker_now as _broker_now

        term = str(_brokers.terminal_path("darwinex"))
        if not mt5.initialize(path=term):
            raise RuntimeError(f"mt5.initialize failed — terminal not running at {term}")
        try:
            broker_dt = _broker_now()
        finally:
            mt5.shutdown()

        if broker_dt is None:
            raise RuntimeError("broker_now() returned None — no EURUSD tick")

        host_epoch = _time_module.time()
        # broker_now() = datetime(1970,1,1) + timedelta(seconds=tick.time)
        broker_epoch = (broker_dt - datetime(1970, 1, 1)).total_seconds()
        raw_diff_s = host_epoch - broker_epoch

        # Expected EET window: raw_diff ∈ [-10800, -7200] ± 1800 s tolerance
        # = [-12600, -5400].  Outside this range → anomalous clock drift.
        eet_low = -10800 - 1800   # -12600
        eet_high = -7200  + 1800  # -5400
        drift_h = raw_diff_s / 3600.0
        detail = (
            f"host_epoch={host_epoch:.0f}  broker_epoch={broker_epoch:.0f}  "
            f"raw_diff={drift_h:+.2f}h  (expected −2h to −3h for EET)"
        )
        if raw_diff_s < eet_low or raw_diff_s > eet_high:
            logger.warning("CLOCK DRIFT anomaly: %s", detail)
            _send_alert(f"[daily-chain] clock_drift WARNING: {detail}")
            code = 1
        else:
            logger.info("clock_drift OK: %s", detail)
            code = 0

    except Exception as exc:  # noqa: BLE001
        # Terminal not running, MT5 not installed, etc. — non-fatal, skip silently.
        detail = f"skipped ({exc})"
        code = 0
        logger.info("clock_drift: %s", detail)

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    _record_finish(conn, run_id, code, detail)
    return StepResult("clock_drift", code, elapsed, detail)


# ── Step d: signal refresh (run as -c subprocess for crash isolation) ─────────

# Single-line command passed to `python -c`.  Sets PYTHONPATH before import so
# bootstrap_runtime can locate the repo root even without an installed package.
_SIGNAL_REFRESH_CMD = (
    "from lib.core.runtime_bootstrap import bootstrap_runtime; bootstrap_runtime(); "
    "from deployment.live.broker_data import bind_signal_cache, refresh_signal_daily; "
    "from deployment.live.forecast_engine import ForecastEngineConfig, VaultForecastEngine; "
    "engine = VaultForecastEngine(ForecastEngineConfig(vault_root='vault')).load(); "
    "store = bind_signal_cache(); "
    "report = refresh_signal_daily("
    "    engine.required_tickers, store=store, vault_root='vault', populate_bias=True"
    "); "
    "warm = report.warm(500); cold = report.cold(500); "
    "print(f'signal_refresh: warm={list(warm)} cold={list(cold)}')"
)


# ── Chain orchestrator ────────────────────────────────────────────────────────


def run_chain(steps_to_run: frozenset[str]) -> list[StepResult]:
    """Run each step if it is in ``steps_to_run``; otherwise record it as skipped."""
    results: list[StepResult] = []

    def _skip(name: str) -> StepResult:
        logger.info("STEP %s skipped", name)
        return StepResult(name, -1, 0.0, "skipped")

    def _active(name: str) -> bool:
        return name in steps_to_run

    # a) clock_drift — inline Python (MT5 optional; skip if terminal absent)
    results.append(step_clock_drift() if _active("clock_drift") else _skip("clock_drift"))

    # b) MT5 M1 bar scrape
    results.append(
        _run_step_subprocess("scrape", [_VENV_PYTHON, "-m", "data_platform.providers.mt5.scraper"])
        if _active("scrape") else _skip("scrape")
    )

    # c) data-freshness tripwire
    results.append(
        _run_step_subprocess("freshness", [_VENV_PYTHON, "-m", "deployment.ops.check_data_freshness"])
        if _active("freshness") else _skip("freshness")
    )

    # d) shared Darwinex signal cache rebuild (reads scraped parquet, no MT5 connection)
    results.append(
        _run_step_subprocess("signal_refresh", [_VENV_PYTHON, "-c", _SIGNAL_REFRESH_CMD])
        if _active("signal_refresh") else _skip("signal_refresh")
    )

    # e) Nautilus catalog incremental append — strictly-forward, no skip_disjoint_check.
    # Skipped gracefully when the catalog is empty (the materialize script returns 0
    # with "not yet materialized" skip messages, which is an exit code of 0).
    # Runs BEFORE registry_ingest so the registry's catalog scan sees fresh files.
    results.append(
        _run_step_subprocess(
            "catalog_append",
            [_VENV_PYTHON, "-m", "data_platform.nautilus.materialize", "--append"],
        )
        if _active("catalog_append") else _skip("catalog_append")
    )

    # e1) registry ingest live
    results.append(
        _run_step_subprocess(
            "registry_ingest",
            [_VENV_PYTHON, "-m", "data_platform.registry", "ingest", "live"],
        )
        if _active("registry_ingest") else _skip("registry_ingest")
    )

    # e2) registry backup
    results.append(
        _run_step_subprocess(
            "registry_backup",
            [_VENV_PYTHON, "-m", "data_platform.registry", "backup"],
        )
        if _active("registry_backup") else _skip("registry_backup")
    )

    return results


# ── Summary + alert ───────────────────────────────────────────────────────────


def _summarize(results: list[StepResult]) -> int:
    """Print step summary; send Telegram alert for failures; return 0 if all OK."""
    print("\n=== daily-chain summary ===")
    failed: list[StepResult] = []
    for r in results:
        tag = "SKIP" if r.exit_code == -1 else ("OK  " if r.exit_code == 0 else "FAIL")
        print(f"  {tag}  {r.name:<20}  {r.duration_s:5.1f}s  {r.detail}")
        if r.exit_code not in (0, -1):
            failed.append(r)

    if failed:
        names = ", ".join(r.name for r in failed)
        lines = [f"[daily-chain] FAILED steps: {names}"] + [
            f"  {r.name}: exit={r.exit_code}  {r.detail}" for r in failed
        ]
        msg = "\n".join(lines)
        logger.error(msg)
        _send_alert(msg)
        return 1

    print("All steps OK.")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the daily data-platform maintenance chain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Valid step names: {', '.join(ALL_STEPS)}\n\n"
            "Examples:\n"
            "  python -m deployment.ops.run_daily_chain\n"
            "  python -m deployment.ops.run_daily_chain --only scrape\n"
            "  python -m deployment.ops.run_daily_chain --skip clock_drift,freshness\n"
            "  python -m deployment.ops.run_daily_chain --only catalog_append"
        ),
    )
    exc_group = parser.add_mutually_exclusive_group()
    exc_group.add_argument(
        "--skip",
        metavar="STEP[,STEP]",
        help="Comma-separated step names to skip.",
    )
    exc_group.add_argument(
        "--only",
        metavar="STEP",
        help="Run only this single step (all others are skipped).",
    )
    args = parser.parse_args()

    if args.only:
        if args.only not in ALL_STEPS:
            print(
                f"Unknown step {args.only!r}. Valid: {', '.join(ALL_STEPS)}",
                file=sys.stderr,
            )
            return 2
        steps_to_run = frozenset([args.only])

    elif args.skip:
        skipped = {s.strip() for s in args.skip.split(",") if s.strip()}
        unknown = skipped - set(ALL_STEPS)
        if unknown:
            print(
                f"Unknown step(s) in --skip: {unknown}. Valid: {', '.join(ALL_STEPS)}",
                file=sys.stderr,
            )
            return 2
        steps_to_run = frozenset(ALL_STEPS) - skipped

    else:
        steps_to_run = frozenset(ALL_STEPS)

    results = run_chain(steps_to_run)
    return _summarize(results)


if __name__ == "__main__":
    raise SystemExit(main())
