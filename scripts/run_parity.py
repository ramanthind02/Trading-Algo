"""One-line convenience runner for the research parity gate (WP-1).

Other work packages run this to confirm they did not drift research outputs.

Usage (from repo root)::

    # Verify current outputs against committed golden snapshots (the gate):
    .\\.venv\\Scripts\\python.exe scripts\\run_parity.py
    .\\.venv\\Scripts\\python.exe scripts\\run_parity.py --verify

    # Regenerate snapshots (ONLY for intentional, reviewed behaviour changes):
    .\\.venv\\Scripts\\python.exe scripts\\run_parity.py --regen

    # Restrict to one case:
    .\\.venv\\Scripts\\python.exe scripts\\run_parity.py -k feature_research

This wrapper just shells out to pytest with the right marker expression so callers
do not have to remember ``-m regen`` vs the verify path. It uses the *current*
Python interpreter (invoke it with the repo venv interpreter, per CLAUDE.md).

A non-zero exit code means a parity failure (or a hard error) — that BLOCKS the
change. A skip (e.g. environment cannot import the pipelines) exits 0 but prints
the skip reason; read it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parents[1] / "tests" / "parity"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--verify",
        action="store_true",
        help="Verify against golden snapshots (default).",
    )
    group.add_argument(
        "--regen",
        action="store_true",
        help="Regenerate golden snapshots (intentional changes only).",
    )
    parser.add_argument("-k", dest="keyword", default=None, help="pytest -k filter.")
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args passed straight through to pytest.",
    )
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "pytest", str(PARITY_DIR), "-v"]
    if args.regen:
        cmd += ["-m", "regen"]
    else:
        # Verify path: exclude the snapshot-writing tests.
        cmd += ["-m", "parity and not regen"]
    if args.keyword:
        cmd += ["-k", args.keyword]
    if args.extra:
        # argparse.REMAINDER keeps a leading '--'; drop it if present.
        extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
        cmd += extra

    print("parity:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
