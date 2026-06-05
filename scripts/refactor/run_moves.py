"""WP-8 ``git mv`` runner — one batch at a time, leaf-first.

Performs the path moves from ``move_map.PATH_MOVES`` for a single batch, in the
leaf-first order from ``wp8_move_map.md`` §6. It does NOT run automatically and
NEVER touches more than the selected batch.

Usage
-----
    python run_moves.py --list
    python run_moves.py --batch B2 --dry-run
    python run_moves.py --batch B2 --apply
    python run_moves.py --all --dry-run        # preview every batch's moves

Safety
------
* ``--dry-run`` is the default. ``--apply`` is required to execute ``git mv``.
* Optional batches (B11 prop_firms, B12 nodes->signals) are SKIPPED unless named
  explicitly with ``--batch`` (``--all`` includes them only with
  ``--include-optional``).
* Each move's old_path is resolved against the working tree; a move whose source
  is missing is REPORTED and skipped (it is not an error — e.g. probes culled in
  Phase 1, or directory moves already partially done).
* Parent directories of new_path are created with ``git mv`` semantics: git mv
  into a not-yet-existing dir requires the dir to exist, so we ``mkdir -p`` the
  parent first (a no-op in --dry-run).
* After a directory move, the operator runs ``rewrite_imports.py --apply`` then
  the parity gate (see README). This script does only the ``git mv`` step.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import move_map as mm  # noqa: E402


def _repo_root(arg: Path | None) -> Path:
    return (arg or Path(__file__).resolve().parents[2]).resolve()


def _moves_for_batch(batch: str) -> list[mm.PathMove]:
    return [m for m in mm.PATH_MOVES if m.batch == batch]


def _run_git_mv(root: Path, old: str, new: str, *, apply: bool) -> str:
    src = root / old
    dst = root / new
    if not src.exists():
        return f"  MISSING  {old}  (source absent — skipped)"
    if dst.exists():
        return f"  EXISTS   {new}  (destination present — skipped)"
    if not apply:
        return f"  [dry-run] git mv {old} -> {new}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "mv", old, new],
        cwd=root, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return f"  FAILED   git mv {old} -> {new}: {result.stderr.strip()}"
    return f"  moved    {old} -> {new}"


def _print_batches() -> None:
    print("Batches (leaf-first; B0/B13 are not move batches):")
    for b in mm.BATCH_ORDER:
        n = len(_moves_for_batch(b.name))
        flag = " [OPTIONAL]" if b.optional else ""
        moves = f"{n} move(s)" if b.has_moves else "no moves (manual)"
        print(f"  {b.name:4s} {moves:18s}{flag}  {b.summary}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WP-8 git mv runner (per batch)")
    p.add_argument("--list", action="store_true", help="list batches and exit")
    p.add_argument("--batch", help="batch name to run (e.g. B2)")
    p.add_argument("--all", action="store_true",
                   help="preview/run every move batch in order")
    p.add_argument("--include-optional", action="store_true",
                   help="with --all, include optional batches B11/B12")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True,
                      help="preview only (default)")
    mode.add_argument("--apply", action="store_true", help="execute git mv")
    p.add_argument("--root", type=Path, default=None)
    args = p.parse_args(argv)

    if args.list:
        _print_batches()
        return 0

    if not args.batch and not args.all:
        p.error("specify --batch <name> or --all (or --list)")

    root = _repo_root(args.root)
    apply = bool(args.apply)

    if args.batch:
        batches = [args.batch]
    else:
        batches = [
            b.name for b in mm.BATCH_ORDER
            if b.has_moves and (args.include_optional or not b.optional)
        ]

    mode_str = "APPLY" if apply else "DRY-RUN"
    print(f"=== run_moves [{mode_str}] root={root} ===")
    total_planned = total_missing = total_moved = 0

    for batch in batches:
        if batch not in mm.BATCH_NAMES:
            print(f"Unknown batch '{batch}'. Use --list.", file=sys.stderr)
            return 2
        moves = _moves_for_batch(batch)
        binfo = next(b for b in mm.BATCH_ORDER if b.name == batch)
        opt = " [OPTIONAL]" if binfo.optional else ""
        print(f"\n-- {batch}{opt}: {binfo.summary} ({len(moves)} move(s)) --")
        if not moves:
            print("  (no path moves in this batch)")
            continue
        for m in moves:
            line = _run_git_mv(root, m.old_path, m.new_path, apply=apply)
            if "MISSING" in line or "EXISTS" in line:
                total_missing += 1
            elif line.lstrip().startswith("moved"):
                total_moved += 1
            else:
                total_planned += 1
            if m.note:
                line += f"   # {m.note}"
            print(line)

    print(f"\nSummary: planned={total_planned} moved={total_moved} missing={total_missing}")
    if not apply:
        print("Nothing was changed (dry-run). Re-run with --apply to execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
