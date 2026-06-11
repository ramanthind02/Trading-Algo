"""One-shot migration: converge all 214 D1 bar files onto MT5_D1_BARS_SCHEMA
(int64 tick_volume / int16 spread) — ADR-6 amended 2026-06-09, M0.3.

There are two source populations:
- 204 files already at int32 tick_volume / int16 spread (previous migration target
  MT5_BARS_SCHEMA): widen tick_volume int32→int64 (lossless — no overflow possible).
- 10 legacy US-stock files at int64 tick_volume / int32 spread, no metadata:
  narrow spread int32→int16 (assert fits first — hard error naming the file if not),
  add timezone stamp.

Usage::

    python -m data_platform.providers.mt5.d1_schema_migration            # all symbols
    python -m data_platform.providers.mt5.d1_schema_migration --dry-run
    python -m data_platform.providers.mt5.d1_schema_migration --symbols BAC EURUSD

Per-file logic
--------------
1. Read the file with pyarrow (no schema cast — raw on-disk types).
2. If tick_volume==int64 AND spread==int16 AND timezone metadata present
   → mark 'already-canonical', skip.
3. Assert max(spread) fits int16 (hard error naming the file if not);
   tick_volume widening (int32→int64) is always lossless — no assertion needed.
4. Cast to MT5_D1_BARS_SCHEMA.
5. Parity check: to_pandas() of old vs new must be equal after upcasting any
   narrowed columns back to the old wider dtypes — i.e. values unchanged.
6. Write via write_mt5_d1_bars (atomic).

--dry-run reports per-file verdicts without writing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _HERE.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
    _HERE.parents[3],
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_platform.storage.contracts import MT5_D1_BARS_SCHEMA  # noqa: E402
from data_platform.storage import write_mt5_d1_bars  # noqa: E402

_MT5_DATA_DIR = _REPO_ROOT / "data" / "mt5_data"
_INT16_MAX = 32_767


def _migrate_file(path: Path, *, dry_run: bool) -> str:
    """Migrate one D1 parquet file in-place (atomic write).

    Returns one of:
      ``'already-canonical'`` — schema fields already match; file untouched.
      ``'migrated'``          — successfully cast and (if not dry_run) written.
      ``'failed:<reason>'``   — overflow, parity, or I/O error; file untouched.
    """
    try:
        table = pq.read_table(path)
    except Exception as exc:
        return f"failed:read error — {exc}"

    # A file is "already-canonical" when:
    #   tick_volume == int64  (amended ADR-6: daily stock volumes exceed int32)
    #   spread      == int16
    #   timezone metadata present (ADR-2 stamp)
    # PyArrow always writes timestamp[s] back as timestamp[ms] on disk, so we
    # cannot use full fields_equal — it would never be True.
    schema_meta = table.schema.metadata or {}
    tv_ok = table.schema.field("tick_volume").type == pa.int64()
    sp_ok = table.schema.field("spread").type == pa.int16()
    tz_ok = schema_meta.get(b"timezone") == b"broker_eet_as_utc"
    if tv_ok and sp_ok and tz_ok:
        return "already-canonical"

    # -- Overflow assertions (only spread narrowing can overflow) --
    # tick_volume widening (int32→int64 or int64→int64) is always lossless;
    # no assertion needed for tick_volume.
    try:
        sp_i64 = table.column("spread").cast(pa.int64())
        max_sp = pc.max(sp_i64).as_py()
        min_sp = pc.min(sp_i64).as_py()
        if max_sp is not None and max_sp > _INT16_MAX:
            return f"failed:spread overflow — max={max_sp} > int16 max {_INT16_MAX}"
        if min_sp is not None and min_sp < 0:
            return f"failed:spread negative — min={min_sp}"
    except Exception as exc:
        return f"failed:overflow check error — {exc}"

    # -- Cast to canonical D1 schema --
    try:
        new_table = table.cast(MT5_D1_BARS_SCHEMA)
    except Exception as exc:
        return f"failed:cast error — {exc}"

    # -- Parity check: values must be unchanged --
    try:
        old_df = table.to_pandas()
        new_df = new_table.to_pandas()

        # Time column may change resolution (ms→s); compare at second precision.
        # D1 midnight bars have no sub-second component so the cast is lossless.
        old_t = old_df["time"].dt.as_unit("s")
        new_t = new_df["time"].dt.as_unit("s")
        if not old_t.equals(new_t):
            return "failed:parity check — time values changed after cast"

        # Upcast the narrowed columns back to the old (wider) dtypes for comparison.
        check_df = new_df.copy()
        check_df["tick_volume"] = check_df["tick_volume"].astype(old_df["tick_volume"].dtype)
        check_df["spread"] = check_df["spread"].astype(old_df["spread"].dtype)
        other_cols = [c for c in old_df.columns if c != "time"]
        if not old_df[other_cols].equals(check_df[other_cols]):
            diff = [c for c in other_cols if not old_df[c].equals(check_df[c])]
            return f"failed:parity check — values changed in columns {diff}"
    except Exception as exc:
        return f"failed:parity check error — {exc}"

    if dry_run:
        return "migrated"  # report only; do not write

    # -- Atomic write through the gated D1 writer --
    try:
        write_mt5_d1_bars(new_table, path)
    except Exception as exc:
        return f"failed:write error — {exc}"

    return "migrated"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Converge D1 bar store onto MT5_D1_BARS_SCHEMA (amended ADR-6)."
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report per-file verdicts without writing.",
    )
    p.add_argument(
        "--symbols", nargs="*", default=None,
        help="Restrict to these symbol names (default: all).",
    )
    args = p.parse_args(argv)

    # Enumerate symbol dirs without recursing; check each explicitly.
    sym_dirs = sorted(
        d for d in _MT5_DATA_DIR.iterdir()
        if d.is_dir() and (d / "bars_D1" / "part.parquet").exists()
    )

    if args.symbols:
        filter_set = set(args.symbols)
        sym_dirs = [d for d in sym_dirs if d.name in filter_set]

    total = len(sym_dirs)
    migrated = already_canonical = failed = 0
    failures: list[tuple[str, str]] = []

    label = "DRY RUN — " if args.dry_run else ""
    print(f"D1 schema migration (amended ADR-6) — {label}{total} symbols to check.")

    for sym_dir in sym_dirs:
        path = sym_dir / "bars_D1" / "part.parquet"
        verdict = _migrate_file(path, dry_run=args.dry_run)
        if verdict == "already-canonical":
            already_canonical += 1
        elif verdict == "migrated":
            migrated += 1
        else:
            failed += 1
            failures.append((sym_dir.name, verdict))
            print(f"  FAIL {sym_dir.name}: {verdict}")

    print()
    print(
        f"Summary: total={total}  migrated={migrated}  "
        f"already-canonical={already_canonical}  failed={failed}"
    )

    if failures:
        print("\nFailed files:")
        for sym, msg in failures:
            print(f"  {sym}: {msg}")


if __name__ == "__main__":
    main()
