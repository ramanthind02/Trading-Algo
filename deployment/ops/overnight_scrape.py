"""
Overnight MT5 data scrape orchestrator.

Runs two phases sequentially (MT5 IPC allows only one Python process):

  Phase 1 — Rollover tick scraper
    data/mt5_data/{SYM}/ticks_rollover_exit/year=YYYY/part.parquet
    data/mt5_data/{SYM}/ticks_rollover_entry/year=YYYY/part.parquet
    Window: 16:00-16:59 NY (exit) and 18:00-19:00 NY (entry)
    All 844 Darwinex symbols, 2022-01-01 → present

  Phase 2 — M1 bar scraper
    data/mt5_data/{SYM}/bars_M1/year=YYYY/part.parquet
    All 844 Darwinex symbols, 2022-01-01 → present

Run:
  .\.venv\Scripts\python.exe deployment\ops\overnight_scrape.py
  .\.venv\Scripts\python.exe deployment\ops\overnight_scrape.py --phase 1   # rollover only
  .\.venv\Scripts\python.exe deployment\ops\overnight_scrape.py --phase 2   # M1 bars only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# deployment/ops/overnight_scrape.py -> repo root is three levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR    = _REPO_ROOT / "logs"
PYTHON     = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"

FROM_DATE = "2022-01-01"
WORKERS   = "4"


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def run_phase(label: str, module: str, extra_args: list[str], log_path: Path) -> int:
    """Run a scraper module, tee output to log_path. Returns exit code."""
    cmd = [str(PYTHON), "-m", module] + extra_args
    _log(f"START  {label}")
    _log(f"CMD    {' '.join(cmd)}")
    _log(f"LOG    {log_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            print(line, end="", flush=True)
        proc.wait()

    elapsed = time.perf_counter() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    _log(f"END    {label} — {status} — elapsed {h:02d}h{m:02d}m{s:02d}s")
    return proc.returncode


def main() -> None:
    p = argparse.ArgumentParser(description="Overnight MT5 scrape orchestrator")
    p.add_argument("--phase", type=int, choices=[1, 2], default=None,
                   help="Run only phase 1 (rollover ticks) or phase 2 (M1 bars). "
                        "Default: run both sequentially.")
    args = p.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    _log("=" * 70)
    _log("Overnight MT5 scrape — START")
    _log(f"From: {FROM_DATE}  Workers: {WORKERS}")
    if args.phase:
        _log(f"Running phase {args.phase} only")
    _log("=" * 70)

    exit_codes = []

    # ── Phase 1: Rollover tick scraper ───────────────────────────────────────
    if args.phase in (None, 1):
        rc = run_phase(
            label="Phase 1: rollover ticks (exit 16:00-16:59 + entry 18:00-19:00 NY)",
            module="data_platform.providers.mt5.rollover_tick_scraper",
            extra_args=["--from", FROM_DATE, "--workers", WORKERS],
            log_path=LOG_DIR / f"rollover_ticks_{ts}.log",
        )
        exit_codes.append(rc)
        if rc != 0:
            _log("Phase 1 failed — stopping. Re-run with --phase 1 to retry.")
            sys.exit(rc)

    # ── Phase 2: M1 bar scraper ───────────────────────────────────────────────
    if args.phase in (None, 2):
        rc = run_phase(
            label="Phase 2: M1 bars (all 844 symbols)",
            module="data_platform.providers.mt5.scraper",
            extra_args=["--from", FROM_DATE, "--workers", WORKERS],
            log_path=LOG_DIR / f"m1_bars_{ts}.log",
        )
        exit_codes.append(rc)

    _log("=" * 70)
    _log(f"Overnight scrape COMPLETE — phase exit codes: {exit_codes}")
    _log("=" * 70)
    sys.exit(max(exit_codes) if exit_codes else 0)


if __name__ == "__main__":
    main()
