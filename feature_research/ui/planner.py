"""Pure helpers for the unified feature_research workspace UI."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import re
import shlex
from typing import Final, Iterable

from feature_research._internal.bias_spec_catalog import first_bias_spec
from feature_research.config import (
    ResearchConfig,
    ResearchWindowConfig,
    ephemeral_weight_hierarchy_group_for_tickers,
)
from feature_research.exploration.pass1_vol_regime import exploration_includes_filter_gate
from feature_research.shared import FeatureResearchPhase
from feature_research.shared.visualization_paths import canonical_in_sample_visualization_dir
from feature_research.ui.combo_analysis import build_combo_diagnostics
from feature_research.ui.contracts import (
    ComboDiagnostics,
    FeatureResearchUiDefaults,
    FeatureResearchUiRequest,
    PhaseOption,
    PhaseRunPlan,
    PhaseWindowOverride,
    PlanField,
    UiRunAction,
)
from feature_research.ui.module_catalog import (
    UiModuleSelection,
    configured_module_selection,
    module_description_for_name,
)
from feature_research.ui.workflow_steps import workflow_steps
from utils.core.enums import Ticker
from utils.evaluation.walkforward.io import resolve_walkforward_output_dir

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_WINDOW_FIELDS: Final[tuple[str, ...]] = (
    "train_start",
    "train_end",
    "val_start",
    "val_end",
)
_PHASE_OPTIONS: Final[tuple[PhaseOption, ...]] = (
    PhaseOption(
        phase=FeatureResearchPhase.EXPLORATION,
        label="Exploration",
        description="Grid search, robustness checks, and in-sample research exports.",
    ),
    PhaseOption(
        phase=FeatureResearchPhase.VALIDATION,
        label="Validation",
        description="One validation walkforward fold using the selected evaluation combo.",
    ),
    PhaseOption(
        phase=FeatureResearchPhase.PORTFOLIO_ADDITION,
        label="Portfolio Addition",
        description="Portfolio admission check using combined train + validation data only.",
    ),
)
_PHASE_DESCRIPTIONS: Final[dict[FeatureResearchPhase, str]] = {
    FeatureResearchPhase.EXPLORATION: (
        "Runs the config-defined exploration search space, then writes robustness and visualization artifacts."
    ),
    FeatureResearchPhase.VALIDATION: (
        "Runs one validation fold using the config-defined evaluation combo for the chosen tickers."
    ),
    FeatureResearchPhase.PORTFOLIO_ADDITION: (
        "Runs one portfolio-addition gate using combined train + validation data only."
    ),
}
_PHASE_STEPS: Final[dict[FeatureResearchPhase, tuple[str, ...]]] = {
    FeatureResearchPhase.EXPLORATION: (
        "Read the configured exploration search space from feature_research/config.py.",
        "Run in-sample EDA for every loaded combination.",
        "Write robustness summaries (DSR, rolling IS, optional full-grid permutation).",
        "Run vector-shuffle permutation and write summaries when enabled.",
        "Write exploration visualization CSVs for Matplotlib review.",
    ),
    FeatureResearchPhase.VALIDATION: (
        "Use the configured evaluation combo from feature_research/config.py.",
        "Run the validation train/val fold for the selected tickers.",
        "Write walkforward artifacts plus validation visualization outputs.",
    ),
    FeatureResearchPhase.PORTFOLIO_ADDITION: (
        "Use the configured evaluation combo from feature_research/config.py.",
        "Review portfolio addition gate results produced by the Validation run.",
        "Gate artifacts live under visualization/validation alongside validation robustness.",
    ),
}


def _exploration_phase_steps(config: ResearchConfig) -> tuple[str, ...]:
    steps: list[str] = []
    if exploration_includes_filter_gate(config):
        steps.append(
            "Pass 1 — ATR% decile binning on the ungated signal: mean return by vol decile "
            "(atr_pct_decile_chart.png; informs low vs high ATR% filter tails in the auto A/B/C grid)."
        )
    gates = getattr(config, "exploration_filter_gates", None)
    if gates is not None and gates.enabled:
        if gates.scope == "winning_signal_only":
            steps.append(
                "After robustness picks the best signal combo, run filter A/B/C comparison "
                "(baseline + SMA252 + ATR low/high, ~15 combos) on that winner only."
            )
        else:
            steps.append(
                "Exploration auto-wraps every signal combo with filter gates (~15× the signal grid)."
            )
    steps.extend(_PHASE_STEPS[FeatureResearchPhase.EXPLORATION])
    return tuple(steps)


def build_ui_defaults(config: ResearchConfig) -> FeatureResearchUiDefaults:
    """Return initial form defaults derived from the current config."""

    selection = configured_module_selection(config)
    return FeatureResearchUiDefaults(
        default_phase=FeatureResearchPhase.EXPLORATION,
        default_tickers=tuple(config.tickers),
        ticker_options=tuple(ticker.name for ticker in Ticker),
        phase_options=_PHASE_OPTIONS,
        workflow_steps=workflow_steps(),
        configured_module=selection.display_label,
        configured_module_description=module_description_for_name(selection.display_label),
        configured_search_space=selection.params_summary,
        config_source_path="feature_research/config.py",
        research_window=_window_override_from_config(config.research_window),
    )


def build_ui_request(
    *,
    phase_name: str,
    tickers_text: str | None,
    fallback_tickers: tuple[Ticker, ...],
    portfolio_tickers_text: str | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    val_start: str | None = None,
    val_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
) -> FeatureResearchUiRequest:
    """Normalize text inputs from the UI into a typed request."""

    phase = FeatureResearchPhase(str(phase_name).strip())
    window_override = _parse_window_override(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )
    if phase is FeatureResearchPhase.EXPLORATION and window_override is not None:
        raise ValueError("Exploration window overrides are not supported.")
    portfolio_tickers = _parse_optional_tickers_text(portfolio_tickers_text)
    return FeatureResearchUiRequest(
        phase=phase,
        tickers=_parse_tickers_text(tickers_text, fallback_tickers),
        portfolio_tickers=portfolio_tickers,
        window_override=window_override,
    )


def apply_ui_request(
    config: ResearchConfig,
    request: FeatureResearchUiRequest,
) -> ResearchConfig:
    """Apply all UI-driven overrides to a research config."""

    configured, _selection = resolve_ui_config(config, request)
    return configured


def resolve_ui_config(
    config: ResearchConfig,
    request: FeatureResearchUiRequest,
) -> tuple[ResearchConfig, UiModuleSelection]:
    """Return the config and read-only config-backed bias selection.

    ``request.tickers`` (CLI ``--tickers`` or UI selection) overrides
    :attr:`ResearchConfig.tickers`, portfolio inclusion candidates, and vault save
    tickers. Portfolio gate baselines keep ``portfolio_source.tickers`` from
    ``load_config()`` unless ``request.portfolio_tickers`` (``--portfolio-tickers``)
    is set explicitly.
    """

    selection = configured_module_selection(config)
    configured = config
    tickers = list(request.tickers)
    request_ticker_tuple = tuple(request.tickers)
    candidate_ticker_tuple = request_ticker_tuple
    vault_ticker_tuple = request_ticker_tuple
    portfolio_source = configured.portfolio_source
    if portfolio_source is not None and request.portfolio_tickers is not None:
        portfolio_source = replace(
            portfolio_source,
            tickers=tuple(request.portfolio_tickers),
        )
    portfolio_inclusion = (
        None
        if configured.portfolio_inclusion is None
        else replace(
            configured.portfolio_inclusion,
            candidate_tickers=candidate_ticker_tuple,
            ephemeral_weight_hierarchy_group=ephemeral_weight_hierarchy_group_for_tickers(
                candidate_ticker_tuple,
                configured_group=configured.portfolio_inclusion.ephemeral_weight_hierarchy_group,
            ),
        )
    )
    vault_save = (
        None
        if configured.vault_save is None
        else replace(configured.vault_save, tickers=vault_ticker_tuple)
    )
    configured = replace(
        configured,
        tickers=tickers,
        portfolio_source=portfolio_source,
        portfolio_inclusion=portfolio_inclusion,
        vault_save=vault_save,
    )
    if request.window_override is None:
        return configured, selection
    _validate_window_against_data_range(configured, request.window_override)
    window = ResearchWindowConfig(
        train_start=request.window_override.train_start,
        train_end=request.window_override.train_end,
        val_start=request.window_override.val_start,
        val_end=request.window_override.val_end,
    )
    match request.phase:
        case FeatureResearchPhase.EXPLORATION:
            raise ValueError("Exploration window overrides are not supported.")
        case FeatureResearchPhase.VALIDATION | FeatureResearchPhase.PORTFOLIO_ADDITION:
            return replace(configured, research_window=window), selection


def build_phase_plan(
    config: ResearchConfig,
    request: FeatureResearchUiRequest,
) -> PhaseRunPlan:
    """Build a rendered preview for the requested phase run."""

    configured, selection = resolve_ui_config(config, request)
    combo_diagnostics = build_combo_diagnostics(configured)
    output_path = _phase_output_path(configured, request.phase)
    phase_bias_spec = (
        configured.bias_spec
        if request.phase is FeatureResearchPhase.EXPLORATION
        else configured.eval_bias_spec
    )
    phase_module_name = str(first_bias_spec(phase_bias_spec)["module_name"])
    fields = (
        PlanField(
            label="Research tickers",
            value=", ".join(ticker.name for ticker in request.tickers),
        ),
        PlanField(label="Timeframe", value=configured.timeframe.name),
        PlanField(label="Target", value=configured.target_col),
        PlanField(label="Configured bias node", value=selection.display_label),
        PlanField(label="Phase module", value=phase_module_name),
        PlanField(label="Configured search space", value=selection.params_summary),
        *_combo_fields(combo_diagnostics),
        PlanField(label="Configured data range", value=_format_window(configured.start, configured.end)),
        PlanField(label="Config source", value="feature_research/config.py"),
        *_portfolio_source_fields(configured),
        *_phase_window_fields(configured, request.phase),
        PlanField(label="Output path", value=output_path.as_posix()),
    )
    return PhaseRunPlan(
        phase=request.phase,
        title=f"{_phase_label(request.phase)} workspace plan",
        description=_PHASE_DESCRIPTIONS[request.phase],
        command=build_phase_command(request, UiRunAction.EXECUTE),
        output_path=output_path,
        combo_diagnostics=combo_diagnostics,
        fields=fields,
        will_run=(
            _exploration_phase_steps(configured)
            if request.phase is FeatureResearchPhase.EXPLORATION
            else _PHASE_STEPS[request.phase]
        ),
        notes=_phase_notes(configured, request.phase, combo_diagnostics),
    )


def build_phase_command(
    request: FeatureResearchUiRequest,
    action: UiRunAction,
) -> str:
    """Build the repo-root terminal command for the requested run."""

    python_cmd = _preferred_python_command()
    tokens = [
        python_cmd,
        "-m",
        "feature_research.ui.run_phase",
        "--phase",
        request.phase.value,
        "--tickers",
        ",".join(ticker.name for ticker in request.tickers),
        "--action",
        action.value,
    ]
    if request.portfolio_tickers is not None:
        tokens.extend(
            [
                "--portfolio-tickers",
                ",".join(ticker.name for ticker in request.portfolio_tickers),
            ]
        )
    window_tokens = (
        []
        if request.window_override is None
        else [
            "--train-start",
            request.window_override.train_start.date().isoformat(),
            "--train-end",
            request.window_override.train_end.date().isoformat(),
            "--val-start",
            request.window_override.val_start.date().isoformat(),
            "--val-end",
            request.window_override.val_end.date().isoformat(),
        ]
    )
    return shlex.join([*tokens, *window_tokens])


def defaults_to_dict(defaults: FeatureResearchUiDefaults) -> dict[str, object]:
    """Serialize UI defaults for Flask endpoints."""

    return {
        "default_phase": defaults.default_phase.value,
        "default_tickers": ",".join(ticker.name for ticker in defaults.default_tickers),
        "ticker_options": list(defaults.ticker_options),
        "phase_options": [
            {
                "value": option.phase.value,
                "label": option.label,
                "description": option.description,
            }
            for option in defaults.phase_options
        ],
        "workflow_steps": [
            {
                "phase": step.phase.value,
                "title": step.title,
                "summary": step.summary,
                "gate": step.gate,
                "docs_path": step.docs_path,
            }
            for step in defaults.workflow_steps
        ],
        "configured_module": defaults.configured_module,
        "configured_module_description": defaults.configured_module_description,
        "configured_search_space": defaults.configured_search_space,
        "config_source_path": defaults.config_source_path,
        "research_window": _window_override_to_dict(defaults.research_window),
    }


def plan_to_dict(plan: PhaseRunPlan) -> dict[str, object]:
    """Serialize a phase plan for Flask endpoints."""

    diagnostics = plan.combo_diagnostics
    return {
        "phase": plan.phase.value,
        "title": plan.title,
        "description": plan.description,
        "command": plan.command,
        "output_path": plan.output_path.as_posix(),
        "combo_diagnostics": (
            None
            if diagnostics is None
            else {
                "raw_combo_count": diagnostics.raw_combo_count,
                "loaded_combo_count": diagnostics.loaded_combo_count,
                "effective_combo_count": diagnostics.effective_combo_count,
                "evaluation_combo_label": diagnostics.evaluation_combo_label,
                "surface_interpretation": diagnostics.surface_interpretation,
                "notes": list(diagnostics.notes),
            }
        ),
        "fields": [{"label": field.label, "value": field.value} for field in plan.fields],
        "will_run": list(plan.will_run),
        "notes": list(plan.notes),
    }


def _portfolio_source_fields(config: ResearchConfig) -> tuple[PlanField, ...]:
    source = config.portfolio_source
    if source is None:
        return ()
    if source.ensemble_dirs is not None:
        return (
            PlanField(
                label="Portfolio baseline",
                value=f"{len(source.ensemble_dirs)} explicit ensemble(s)",
            ),
        )
    if source.vault_root is not None:
        return (
            PlanField(label="Portfolio vault root", value=str(source.vault_root)),
        )
    profile = source.vault_profile or "prop"
    ticker_label = (
        ", ".join(ticker.name for ticker in source.tickers)
        if source.tickers
        else "all vault instruments"
    )
    return (
        PlanField(
            label="Portfolio vault",
            value="prop (vault/)" if profile == "prop" else "personal (vault_personal/)",
        ),
        PlanField(label="Portfolio gate tickers", value=ticker_label),
    )


def _combo_fields(combo_diagnostics: ComboDiagnostics) -> tuple[PlanField, ...]:
    effective_text = (
        "Unavailable"
        if combo_diagnostics.effective_combo_count is None
        else f"{combo_diagnostics.effective_combo_count:.2f}"
    )
    return (
        PlanField(label="Raw parameter combos", value=str(combo_diagnostics.raw_combo_count)),
        PlanField(label="Loaded parameter combos", value=str(combo_diagnostics.loaded_combo_count)),
        PlanField(label="Effective parameter combos", value=effective_text),
        PlanField(label="Evaluation combo", value=combo_diagnostics.evaluation_combo_label),
    )


def _parse_optional_tickers_text(tickers_text: str | None) -> tuple[Ticker, ...] | None:
    cleaned = "" if tickers_text is None else str(tickers_text).strip()
    if not cleaned:
        return None
    return _parse_ticker_tokens(
        token.upper()
        for token in re.split(r"[\s,]+", cleaned)
        if token.strip()
    )


def _parse_tickers_text(
    tickers_text: str | None,
    fallback_tickers: tuple[Ticker, ...],
) -> tuple[Ticker, ...]:
    cleaned = "" if tickers_text is None else str(tickers_text).strip()
    tokens = [
        token.upper()
        for token in re.split(r"[\s,]+", cleaned)
        if token.strip()
    ]
    if not tokens:
        return fallback_tickers
    return _parse_ticker_tokens(tokens)


def _parse_ticker_tokens(tokens: Iterable[str]) -> tuple[Ticker, ...]:
    try:
        return tuple(Ticker[token] for token in tokens)
    except KeyError as exc:
        raise ValueError(f"Unknown ticker: {exc.args[0]}") from exc


def _parse_window_override(
    *,
    train_start: str | None,
    train_end: str | None,
    val_start: str | None = None,
    val_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
) -> PhaseWindowOverride | None:
    resolved_val_start = val_start if val_start is not None else test_start
    resolved_val_end = val_end if val_end is not None else test_end
    raw_values = {
        "train_start": _optional_text(train_start),
        "train_end": _optional_text(train_end),
        "val_start": _optional_text(resolved_val_start),
        "val_end": _optional_text(resolved_val_end),
    }
    provided = {name: value for name, value in raw_values.items() if value is not None}
    if not provided:
        return None
    if len(provided) != len(_WINDOW_FIELDS):
        raise ValueError("Fill all four window dates or leave them all blank.")
    return PhaseWindowOverride(
        train_start=_parse_iso_date(raw_values["train_start"], field_name="train_start"),
        train_end=_parse_iso_date(raw_values["train_end"], field_name="train_end"),
        val_start=_parse_iso_date(raw_values["val_start"], field_name="val_start"),
        val_end=_parse_iso_date(raw_values["val_end"], field_name="val_end"),
    )


def _parse_iso_date(value: str | None, *, field_name: str) -> datetime:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.") from exc


def _optional_text(value: str | None) -> str | None:
    cleaned = None if value is None else str(value).strip()
    return cleaned or None


def _validate_window_against_data_range(
    config: ResearchConfig,
    window: PhaseWindowOverride,
) -> None:
    if window.train_start < config.start or window.val_end > config.end:
        raise ValueError(
            "Window override must stay inside the configured data range shown in the plan."
        )


def _phase_output_path(config: ResearchConfig, phase: FeatureResearchPhase) -> Path:
    if phase is FeatureResearchPhase.EXPLORATION:
        return config.reports_dir
    output_subdir = "validation" if phase is FeatureResearchPhase.VALIDATION else "oos"
    return resolve_walkforward_output_dir(
        feature_type=config.feature_type.value,
        module_name=str(config.eval_bias_spec["module_name"]),
        root_dir=config.output_root,
        output_subdir=output_subdir,
    )


def _phase_window_fields(
    config: ResearchConfig,
    phase: FeatureResearchPhase,
) -> tuple[PlanField, ...]:
    if phase is FeatureResearchPhase.EXPLORATION:
        train_start, train_end = config.training_window_bounds
        return (PlanField("Effective training window", _format_window(train_start, train_end)),)
    if config.research_window is None:
        return (PlanField("Research window", "Not configured"),)
    return (
        PlanField(
            "Train window",
            _format_window(config.research_window.train_start, config.research_window.train_end),
        ),
        PlanField(
            "Validation window",
            _format_window(config.research_window.val_start, config.research_window.val_end),
        ),
    )


def _phase_notes(
    config: ResearchConfig,
    phase: FeatureResearchPhase,
    combo_diagnostics: ComboDiagnostics,
) -> tuple[str, ...]:
    preview_note = (
        "The workspace can launch phases directly, and the command remains available for terminal use."
    )
    robustness_note = (
        "Robustness is enabled in the current config."
        if config.robustness.enabled
        else "Robustness is disabled in the current config."
    )
    full_grid_note = (
        "Full-grid search-bias permutation is enabled in the current config."
        if config.robustness.enabled and config.robustness.run_full_grid_permutation
        else "Full-grid search-bias permutation is disabled in the current config."
    )
    permutation_note = (
        "Vector-shuffle permutation runs on the robustness-selected combo only."
        if (
            config.permutation.enabled
            and config.permutation.run_vector_shuffle
            and config.permutation.vector_shuffle_scope.value == "selected_combo"
        )
        else (
            "Vector-shuffle permutation runs on the full param grid."
            if config.permutation.enabled and config.permutation.run_vector_shuffle
            else "Vector-shuffle permutation is disabled in the current config."
        )
    )
    visualization_note = (
        f"Exploration visualization CSVs are written to "
        f"{canonical_in_sample_visualization_dir().as_posix()}."
    )
    notes = [preview_note, *combo_diagnostics.notes]
    if phase is FeatureResearchPhase.EXPLORATION:
        notes.extend([robustness_note, full_grid_note, permutation_note, visualization_note])
    if phase is FeatureResearchPhase.PORTFOLIO_ADDITION:
        notes.append(
            "Portfolio addition gate runs automatically when Validation completes; "
            "this phase reviews gate artifacts under visualization/validation/."
        )
    if combo_diagnostics.surface_interpretation:
        notes.append(combo_diagnostics.surface_interpretation)
    return tuple(notes)


def _window_override_from_config(
    window: ResearchWindowConfig | None,
) -> PhaseWindowOverride | None:
    if window is None:
        return None
    return PhaseWindowOverride(
        train_start=window.train_start,
        train_end=window.train_end,
        val_start=window.val_start,
        val_end=window.val_end,
    )


def _window_override_to_dict(
    window: PhaseWindowOverride | None,
) -> dict[str, str] | None:
    if window is None:
        return None
    return {
        "train_start": window.train_start.date().isoformat(),
        "train_end": window.train_end.date().isoformat(),
        "val_start": window.val_start.date().isoformat(),
        "val_end": window.val_end.date().isoformat(),
    }


def _preferred_python_command() -> str:
    venv_python = _REPO_ROOT / "venv" / "bin" / "python"
    dot_venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return "./venv/bin/python"
    if dot_venv_python.exists():
        return "./.venv/bin/python"
    return "python"


def _phase_label(phase: FeatureResearchPhase) -> str:
    label_lookup = {option.phase: option.label for option in _PHASE_OPTIONS}
    return label_lookup[phase]


def _format_window(start: datetime, end: datetime) -> str:
    return f"{start.date().isoformat()} to {end.date().isoformat()}"
