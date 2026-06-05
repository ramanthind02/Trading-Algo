"""Shared CLI registry for `python -m feature_research`."""
from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, TypeAlias

from research.feature.shared.contracts import FeatureResearchPhase

MainCallable: TypeAlias = Callable[[], int | None]
MainLoader: TypeAlias = Callable[[], MainCallable]
CommandCategory: TypeAlias = Literal["primary", "additional"]


@dataclass(frozen=True)
class FeatureResearchCommand:
    """One CLI command exposed from the package root."""

    name: str
    aliases: tuple[str, ...]
    help_text: str
    loader: MainLoader
    category: CommandCategory


def _load_main(module_path: str) -> MainCallable:
    module = import_module(module_path)
    main_fn = getattr(module, "main", None)
    if main_fn is None or not callable(main_fn):
        raise TypeError(f"{module_path} does not expose a callable main().")
    return main_fn


def _main_loader(module_path: str) -> MainLoader:
    return lambda: _load_main(module_path)


def feature_research_commands() -> tuple[FeatureResearchCommand, ...]:
    """Canonical root commands plus legacy aliases."""

    return (
        FeatureResearchCommand(
            name=FeatureResearchPhase.EXPLORATION.value,
            aliases=("in-sample", "in_sample", "eda"),
            help_text=(
                "Usage: python -m feature_research exploration\n\n"
                "Runs exploration: EDA sweep, optional robustness, optional vector-shuffle permutation."
            ),
            loader=_main_loader("research.feature.in_sample.run_is"),
            category="primary",
        ),
        FeatureResearchCommand(
            name=FeatureResearchPhase.VALIDATION.value,
            aliases=(),
            help_text=(
                "Usage: python -m feature_research validation\n\n"
                "Runs the validation phase walkforward pipeline."
            ),
            loader=_main_loader("research.feature.validation.run_validation"),
            category="primary",
        ),
        FeatureResearchCommand(
            name=FeatureResearchPhase.PORTFOLIO_ADDITION.value,
            aliases=("portfolio-addition", "oos"),
            help_text=(
                "Usage: python -m feature_research portfolio_addition\n\n"
                "Runs the portfolio-addition phase (current out-of-sample pipeline)."
            ),
            loader=_main_loader("research.feature.oos.run_oos"),
            category="primary",
        ),
        FeatureResearchCommand(
            name="binning",
            aliases=(),
            help_text=(
                "Usage: python -m feature_research binning\n\n"
                "Runs the research-only continuous binning exports."
            ),
            loader=_main_loader("research.feature.binning.run_phase"),
            category="additional",
        ),
        FeatureResearchCommand(
            name="oos_permutation",
            aliases=("oos-permutation",),
            help_text=(
                "Usage: python -m feature_research oos_permutation "
                "[--nreps N --seed S --n-jobs N --output-dir PATH]\n\n"
                "Runs the OOS vector-shuffle permutation pipeline."
            ),
            loader=_main_loader("research.feature.oos.run_oos_permutation"),
            category="additional",
        ),
        FeatureResearchCommand(
            name="validation_permutation",
            aliases=("validation-permutation",),
            help_text=(
                "Usage: python -m feature_research validation_permutation "
                "[--nreps N --seed S --n-jobs N --output-dir PATH]\n\n"
                "Runs the validation vector-shuffle permutation pipeline."
            ),
            loader=_main_loader("research.feature.validation.run_validation_permutation"),
            category="additional",
        ),
    )


def command_alias_map(
    commands: Sequence[FeatureResearchCommand] | None = None,
) -> dict[str, FeatureResearchCommand]:
    """Map canonical command names and aliases to their command records."""

    resolved_commands = feature_research_commands() if commands is None else tuple(commands)
    alias_pairs = [
        (alias, command)
        for command in resolved_commands
        for alias in (command.name, *command.aliases)
    ]
    return dict(alias_pairs)


def resolve_command(
    name: str,
    commands: Sequence[FeatureResearchCommand] | None = None,
) -> FeatureResearchCommand | None:
    """Resolve a CLI token to a canonical command record."""

    return command_alias_map(commands).get(name)


def _format_usage_line(command: FeatureResearchCommand) -> str:
    alias_text = (
        ""
        if not command.aliases
        else f" (aliases: {', '.join(command.aliases)})"
    )
    return f"  {command.name}{alias_text}"


def build_usage(commands: Sequence[FeatureResearchCommand] | None = None) -> str:
    """Build the package-root CLI usage text."""

    resolved_commands = feature_research_commands() if commands is None else tuple(commands)
    primary_lines = [
        _format_usage_line(command)
        for command in resolved_commands
        if command.category == "primary"
    ]
    additional_lines = [
        _format_usage_line(command)
        for command in resolved_commands
        if command.category == "additional"
    ]
    sections = [
        "Usage: python -m feature_research <command> [args]",
        "",
        "Primary phases:",
        *primary_lines,
    ]
    extra_sections = (
        ["", "Additional tools:", *additional_lines]
        if additional_lines
        else []
    )
    return "\n".join([*sections, *extra_sections, ""])


def phase_help(
    name: str,
    commands: Sequence[FeatureResearchCommand] | None = None,
) -> str:
    """Return help text for a canonical command or alias."""

    command = resolve_command(name, commands)
    return build_usage(commands) if command is None else command.help_text


def call_main_with_argv(main_fn: MainCallable, argv: Sequence[str]) -> int:
    """Call a phase main() with a temporary `sys.argv`."""

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *argv]
        result = main_fn()
    finally:
        sys.argv = original_argv
    return 0 if result is None else int(result)


def run_command(command: FeatureResearchCommand, argv: Sequence[str]) -> int:
    """Run a resolved command record."""

    return call_main_with_argv(command.loader(), argv)


__all__ = [
    "FeatureResearchCommand",
    "build_usage",
    "call_main_with_argv",
    "command_alias_map",
    "feature_research_commands",
    "phase_help",
    "resolve_command",
    "run_command",
]
