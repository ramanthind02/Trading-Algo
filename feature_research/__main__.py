from __future__ import annotations

import sys
from collections.abc import Sequence

from feature_research.shared.cli import build_usage, phase_help, resolve_command, run_command

_HELP_FLAGS = {"-h", "--help"}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in _HELP_FLAGS:
        print(build_usage())
        return 0 if args else 1

    command_name, command_args = args[0], args[1:]
    if command_args and command_args[0] in _HELP_FLAGS:
        print(phase_help(command_name))
        return 0

    command = resolve_command(command_name)
    if command is None:
        print(f"Unknown command: {command_name}")
        print(build_usage())
        return 1
    return run_command(command, command_args)


if __name__ == "__main__":
    raise SystemExit(main())
