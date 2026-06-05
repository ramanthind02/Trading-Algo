"""CLI entry: ``python -m portfolio_research holdout``."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print("Usage: python -m portfolio_research holdout")
        raise SystemExit(0 if args and args[0] in {"-h", "--help", "help"} else 1)
    if args[0] == "holdout":
        from research.portfolio.holdout.pipeline import main as holdout_main

        holdout_main()
        return
    raise SystemExit(f"Unknown command {args[0]!r}. Try: holdout")


if __name__ == "__main__":
    main()
