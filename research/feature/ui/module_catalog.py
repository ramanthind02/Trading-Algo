"""Curated module catalog and bias-spec builders for the research workspace UI."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from research.feature._internal.bias_spec_catalog import first_bias_spec
from research.feature.config import (
    EvaluationDefaultsCatalog,
    EvaluationPhaseDefaultsConfig,
    FeatureType,
    InSampleDefaultsCatalog,
    InSamplePhaseDefaultsConfig,
    ResearchConfig,
)
from research.feature.in_sample.data_loader import expand_bias_specs
from research.feature.ui.contracts import (
    UiFieldKind,
    UiFieldOption,
    UiModuleOption,
    UiParameterField,
)
from nodes._taxonomy import CANONICAL_MODULE_CLASSES, CANONICAL_MODULE_IMPORTS
from nodes.regime.sma.stacked_sma_long_only import stacked_sma_period_combos
from lib.core.enums import Direction, PositionMode, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parents[1]
_SIGNAL_RESULTS_DIR = _FEATURE_RESEARCH_DIR / "in_sample" / "results" / "signed_signal"
_SUPPORTED_MODULE_NAMES = (
    "stacked_sma_long_only",
    "cyclical_rsi_signal",
    "donchian_long_only",
    "buy_hold",
    "custom",
)


@dataclass(frozen=True)
class UiModuleSelection:
    """Bias specs and metadata derived from a UI module selection."""

    module_name: str
    display_label: str
    exploration_bias_spec: dict[str, Any] | list[dict[str, Any]]
    evaluation_bias_spec: dict[str, Any]
    reports_dir: Path
    params_summary: str


def module_options() -> tuple[UiModuleOption, ...]:
    """User-facing module catalog for the unified workspace."""

    return (
        _catalog_option(
            module_name="stacked_sma_long_only",
            fields=(
                UiParameterField(
                    key="candidate_periods",
                    label="Candidate periods",
                    help_text="Comma-separated SMA periods used to build ordered combinations.",
                    default_value="8,64,128,192,248",
                    placeholder="8,64,128,192,248",
                ),
                UiParameterField(
                    key="layer_count",
                    label="Moving averages",
                    help_text="How many ordered SMA layers to combine.",
                    default_value="4",
                    kind=UiFieldKind.SELECT,
                    options=tuple(
                        UiFieldOption(value=str(value), label=str(value))
                        for value in (2, 3, 4, 5)
                    ),
                ),
            ),
        ),
        _catalog_option(
            module_name="cyclical_rsi_signal",
            fields=(
                UiParameterField(
                    key="short_period",
                    label="Short periods",
                    help_text="Comma-separated short SMA periods.",
                    default_value="2,3,4,5,6",
                    placeholder="2,3,4,5,6",
                ),
                UiParameterField(
                    key="long_period",
                    label="Long periods",
                    help_text="Comma-separated long SMA periods.",
                    default_value="120",
                    placeholder="120",
                ),
                UiParameterField(
                    key="rsi_period",
                    label="RSI periods",
                    help_text="Comma-separated RSI periods.",
                    default_value="2",
                    placeholder="2",
                ),
                UiParameterField(
                    key="oversold",
                    label="Oversold levels",
                    help_text="Comma-separated centered RSI oversold thresholds.",
                    default_value="-20",
                    placeholder="-20",
                ),
                UiParameterField(
                    key="overbought",
                    label="Overbought levels",
                    help_text="Comma-separated centered RSI overbought thresholds.",
                    default_value="20",
                    placeholder="20",
                ),
                UiParameterField(
                    key="strategy_mode",
                    label="Strategy mode",
                    help_text="Choose the directional variant; `long_short` is no longer exposed here.",
                    default_value=Direction.LONG.value,
                    kind=UiFieldKind.SELECT,
                    options=tuple(
                        UiFieldOption(value=value, label=value)
                        for value in ("long", "short")
                    ),
                ),
                UiParameterField(
                    key="exit_policy",
                    label="Exit policy",
                    help_text="Position exit behavior.",
                    default_value="threshold_or_bars",
                    kind=UiFieldKind.SELECT,
                    options=(
                        UiFieldOption(value="threshold_or_bars", label="threshold_or_bars"),
                        UiFieldOption(value="threshold", label="threshold"),
                    ),
                ),
                UiParameterField(
                    key="exit_bars",
                    label="Exit bars",
                    help_text="Comma-separated hold-bar options.",
                    default_value="5",
                    placeholder="5",
                ),
            ),
        ),
        _catalog_option(
            module_name="donchian_long_only",
            fields=(
                UiParameterField(
                    key="channel_lookback",
                    label="Channel lookbacks",
                    help_text="Comma-separated Donchian channel lookbacks.",
                    default_value="20,40,60",
                    placeholder="20,40,60",
                ),
                UiParameterField(
                    key="sma_period",
                    label="SMA filters",
                    help_text="Comma-separated SMA regime values. Use 0 to disable the filter.",
                    default_value="0,200,350",
                    placeholder="0,200,350",
                ),
            ),
        ),
        _catalog_option(module_name="buy_hold", fields=()),
        UiModuleOption(
            value="custom",
            label="custom",
            description="Advanced fallback for arbitrary module names and JSON params.",
            fields=(
                UiParameterField(
                    key="custom_module_name",
                    label="Module name",
                    help_text="Canonical taxonomy key, for example `rsi_signal`.",
                    default_value="stacked_sma_long_only",
                    placeholder="rsi_signal",
                ),
                UiParameterField(
                    key="exploration_params_json",
                    label="Exploration params JSON",
                    help_text="JSON object. Use arrays for grid axes.",
                    default_value=json.dumps(
                        {
                            "period_1": [8, 28, 48],
                            "period_2": [128, 148, 168],
                            "period_3": [188, 208, 228],
                            "period_4": [248],
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    kind=UiFieldKind.TEXTAREA,
                ),
                UiParameterField(
                    key="evaluation_params_json",
                    label="Evaluation params JSON",
                    help_text="Optional JSON override for validation and portfolio-addition.",
                    default_value="",
                    kind=UiFieldKind.TEXTAREA,
                    placeholder='{"period_1": 8, "period_2": 148, "period_3": 208, "period_4": 248}',
                ),
            ),
        ),
    )


def default_module_name(config: ResearchConfig) -> str:
    """Current config module when supported, otherwise `custom`."""

    module_name = str(first_bias_spec(config.bias_spec).get("module_name", "")).strip()
    return module_name if module_name in _SUPPORTED_MODULE_NAMES else "custom"


def default_module_params(config: ResearchConfig) -> dict[str, str]:
    """Best-effort defaults for the current config."""

    module_name = default_module_name(config)
    if module_name == "stacked_sma_long_only":
        return _stacked_sma_defaults_from_config(config)
    if module_name == "cyclical_rsi_signal":
        return _simple_defaults_from_spec(
            first_bias_spec(config.bias_spec),
            keys=(
                "short_period",
                "long_period",
                "rsi_period",
                "oversold",
                "overbought",
                "strategy_mode",
                "exit_policy",
                "exit_bars",
            ),
        )
    if module_name == "donchian_long_only":
        return _simple_defaults_from_spec(
            first_bias_spec(config.bias_spec),
            keys=("channel_lookback", "entry_lookback", "exit_lookback", "sma_period"),
        )
    if module_name == "buy_hold":
        return {}
    spec = first_bias_spec(config.bias_spec)
    evaluation_spec = config.eval_bias_spec
    return {
        "custom_module_name": str(spec.get("module_name", "")).strip(),
        "exploration_params_json": json.dumps(spec.get("params", {}), indent=2, sort_keys=True),
        "evaluation_params_json": json.dumps(
            evaluation_spec.get("params", {}), indent=2, sort_keys=True
        ),
    }


def configured_module_selection(config: ResearchConfig) -> UiModuleSelection:
    """Build a read-only UI summary from the current config bias spec."""

    module_name = str(first_bias_spec(config.bias_spec).get("module_name", "")).strip()
    combo_count = len(expand_bias_specs(config.bias_spec))
    return UiModuleSelection(
        module_name=module_name,
        display_label=module_name,
        exploration_bias_spec=config.bias_spec,
        evaluation_bias_spec=config.eval_bias_spec,
        reports_dir=config.reports_dir,
        params_summary=_configured_search_space_summary(config, combo_count),
    )


def _configured_search_space_summary(config: ResearchConfig, combo_count: int) -> str:
    module_name = str(first_bias_spec(config.bias_spec).get("module_name", "")).strip()
    if module_name in _SUPPORTED_MODULE_NAMES:
        selection = build_module_selection(
            module_name=module_name,
            module_params=default_module_params(config),
            timeframe=config.timeframe,
        )
        return selection.params_summary
    if combo_count == 1:
        return "1 configured combo from feature_research/config.py"
    return f"{combo_count} configured combos from feature_research/config.py"


def module_description_for_name(module_name: str) -> str:
    """Return the taxonomy-backed description when available."""

    return CANONICAL_MODULE_CLASSES.get(
        module_name,
        "Configured directly in feature_research/config.py.",
    )


def build_module_selection(
    *,
    module_name: str,
    module_params: dict[str, str],
    timeframe: TimeFrame,
) -> UiModuleSelection:
    """Translate UI inputs into exploration and evaluation bias specs."""

    normalized_module = module_name.strip()
    if normalized_module == "stacked_sma_long_only":
        return _build_stacked_sma_selection(module_params, timeframe)
    if normalized_module == "cyclical_rsi_signal":
        return _build_cyclical_rsi_selection(module_params, timeframe)
    if normalized_module == "donchian_long_only":
        return _build_donchian_selection(module_params, timeframe)
    if normalized_module == "buy_hold":
        return _build_buy_hold_selection(timeframe)
    if normalized_module == "custom":
        return _build_custom_selection(module_params, timeframe)
    raise ValueError(f"Unsupported module: {module_name}")


def apply_module_selection(
    config: ResearchConfig,
    selection: UiModuleSelection,
) -> ResearchConfig:
    """Return a config whose bias specs are fully controlled by the UI selection."""

    in_sample_defaults = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=selection.exploration_bias_spec,
            target_col=config.target_col,
            strategy=Direction.LONG,
            reports_dir=selection.reports_dir,
            binning_params_overrides={},
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=selection.exploration_bias_spec,
            target_col=config.target_col,
            strategy=Direction.LONG,
            reports_dir=selection.reports_dir,
            binning_params_overrides={},
        ),
    )
    evaluation_defaults = EvaluationDefaultsCatalog(
        continuous=EvaluationPhaseDefaultsConfig(bias_spec=selection.evaluation_bias_spec),
        signed_signal=EvaluationPhaseDefaultsConfig(bias_spec=selection.evaluation_bias_spec),
    )
    return config.__class__(
        tickers=config.tickers,
        start=config.start,
        end=config.end,
        permutation=config.permutation,
        objective_metric_presets=config.objective_metric_presets,
        binning_params=config.binning_params,
        in_sample_defaults=in_sample_defaults,
        timeframe=config.timeframe,
        feature_type=FeatureType.SIGNED_SIGNAL,
        evaluation_defaults=evaluation_defaults,
        param_sensitivity=config.param_sensitivity,
        research_window=config.research_window,
        n_jobs=config.n_jobs,
        output_root=config.output_root,
        generate_ticker_tearsheets=config.generate_ticker_tearsheets,
        tearsheet_target_annual_volatility=config.tearsheet_target_annual_volatility,
        portfolio_source=config.portfolio_source,
        portfolio_inclusion=config.portfolio_inclusion,
        vault_save=config.vault_save,
        robustness=config.robustness,
        sector_allocation_config_path=config.sector_allocation_config_path,
        portfolio_vault_correlation=config.portfolio_vault_correlation,
    )


def module_params_for_defaults(
    module_name: str,
    defaults: dict[str, str] | None = None,
) -> dict[str, str]:
    """Merge provided values onto module defaults."""

    fallback = defaults or {}
    option = next((entry for entry in module_options() if entry.value == module_name), None)
    if option is None:
        raise ValueError(f"Unsupported module: {module_name}")
    field_defaults = {field.key: field.default_value for field in option.fields}
    return {**field_defaults, **fallback}


def _build_stacked_sma_selection(
    module_params: dict[str, str],
    timeframe: TimeFrame,
) -> UiModuleSelection:
    params = module_params_for_defaults("stacked_sma_long_only", module_params)
    candidates = _parse_int_list(params["candidate_periods"], field_name="candidate_periods")
    layer_count = _parse_int_scalar(params["layer_count"], field_name="layer_count", minimum=2)
    combos = stacked_sma_period_combos(layer_count, tuple(candidates))
    if not combos:
        raise ValueError("No stacked SMA combinations were generated.")
    exploration_bias_spec = [
        _stacked_sma_spec(timeframe, combo)
        for combo in combos
    ]
    evaluation_bias_spec = _stacked_sma_spec(timeframe, combos[len(combos) // 2])
    return UiModuleSelection(
        module_name="stacked_sma_long_only",
        display_label="stacked_sma_long_only",
        exploration_bias_spec=exploration_bias_spec,
        evaluation_bias_spec=evaluation_bias_spec,
        reports_dir=_reports_dir_for_module("stacked_sma_long_only", params),
        params_summary=(
            f"{len(combos)} ordered combos from {len(candidates)} candidate periods, "
            f"{layer_count} layers, direction handled by the bias node"
        ),
    )


def _build_cyclical_rsi_selection(
    module_params: dict[str, str],
    timeframe: TimeFrame,
) -> UiModuleSelection:
    params = module_params_for_defaults("cyclical_rsi_signal", module_params)
    short_values = _parse_int_list(params["short_period"], field_name="short_period")
    long_values = _parse_int_list(params["long_period"], field_name="long_period")
    rsi_values = _parse_int_list(params["rsi_period"], field_name="rsi_period")
    oversold_values = _parse_float_list(params["oversold"], field_name="oversold")
    overbought_values = _parse_float_list(params["overbought"], field_name="overbought")
    exit_bars_values = _parse_int_list(params["exit_bars"], field_name="exit_bars", minimum=1)
    strategy_mode = _normalize_ui_strategy_mode(params.get("strategy_mode"))
    exit_policy = params["exit_policy"].strip()
    exploration_params = {
        "short_period": _grid_value(short_values),
        "long_period": _grid_value(long_values),
        "rsi_period": _grid_value(rsi_values),
        "oversold": _grid_value(oversold_values),
        "overbought": _grid_value(overbought_values),
        "strategy_mode": strategy_mode,
        "exit_policy": exit_policy,
        "exit_bars": _grid_value(exit_bars_values),
    }
    evaluation_params = {
        "short_period": _eval_value(short_values),
        "long_period": _eval_value(long_values),
        "rsi_period": _eval_value(rsi_values),
        "oversold": _eval_value(oversold_values),
        "overbought": _eval_value(overbought_values),
        "strategy_mode": strategy_mode,
        "exit_policy": exit_policy,
        "exit_bars": _eval_value(exit_bars_values),
    }
    exploration_bias_spec = {
        "module_name": "cyclical_rsi_signal",
        "timeframes": [timeframe],
        "params": _normalize_direction_params("cyclical_rsi_signal", exploration_params),
    }
    evaluation_bias_spec = {
        "module_name": "cyclical_rsi_signal",
        "timeframes": [timeframe],
        "params": _normalize_direction_params("cyclical_rsi_signal", evaluation_params),
    }
    combo_count = len(expand_bias_specs(exploration_bias_spec))
    return UiModuleSelection(
        module_name="cyclical_rsi_signal",
        display_label="cyclical_rsi_signal",
        exploration_bias_spec=exploration_bias_spec,
        evaluation_bias_spec=evaluation_bias_spec,
        reports_dir=_reports_dir_for_module("cyclical_rsi_signal", params),
        params_summary=(
            f"{combo_count} cyclical RSI combos with exit_policy={exit_policy}"
        ),
    )


def _build_donchian_selection(
    module_params: dict[str, str],
    timeframe: TimeFrame,
) -> UiModuleSelection:
    params = module_params_for_defaults("donchian_long_only", module_params)
    lookbacks = _parse_int_list(params["channel_lookback"], field_name="channel_lookback")
    sma_values = _parse_int_list(
        params["sma_period"],
        field_name="sma_period",
        minimum=0,
        allow_zero=True,
    )
    exploration_bias_spec = {
        "module_name": "donchian_long_only",
        "timeframes": [timeframe],
        "params": {
            "channel_lookback": _grid_value(lookbacks),
            "sma_period": _grid_value(sma_values),
        },
    }
    evaluation_bias_spec = {
        "module_name": "donchian_long_only",
        "timeframes": [timeframe],
        "params": {
            "channel_lookback": _eval_value(lookbacks),
            "sma_period": _eval_value(sma_values),
        },
    }
    combo_count = len(expand_bias_specs(exploration_bias_spec))
    return UiModuleSelection(
        module_name="donchian_long_only",
        display_label="donchian_long_only",
        exploration_bias_spec=exploration_bias_spec,
        evaluation_bias_spec=evaluation_bias_spec,
        reports_dir=_reports_dir_for_module("donchian_long_only", params),
        params_summary=f"{combo_count} Donchian combos across lookback and SMA filter choices",
    )


def _build_buy_hold_selection(timeframe: TimeFrame) -> UiModuleSelection:
    spec = {"module_name": "buy_hold", "timeframes": [timeframe], "params": {}}
    return UiModuleSelection(
        module_name="buy_hold",
        display_label="buy_hold",
        exploration_bias_spec=spec,
        evaluation_bias_spec=spec,
        reports_dir=_reports_dir_for_module("buy_hold", {}),
        params_summary="Single always-long baseline combination",
    )


def _build_custom_selection(
    module_params: dict[str, str],
    timeframe: TimeFrame,
) -> UiModuleSelection:
    params = module_params_for_defaults("custom", module_params)
    custom_module_name = params["custom_module_name"].strip()
    if not custom_module_name:
        raise ValueError("custom_module_name is required.")
    exploration_params = _parse_json_object(
        params["exploration_params_json"],
        field_name="exploration_params_json",
    )
    evaluation_text = params["evaluation_params_json"].strip()
    exploration_bias_spec = {
        "module_name": custom_module_name,
        "timeframes": [timeframe],
        "params": _normalize_direction_params(custom_module_name, exploration_params),
    }
    if evaluation_text:
        evaluation_params = _parse_json_object(
            evaluation_text,
            field_name="evaluation_params_json",
        )
    else:
        evaluation_params = dict(expand_bias_specs(exploration_bias_spec)[0]["params"])
    evaluation_bias_spec = {
        "module_name": custom_module_name,
        "timeframes": [timeframe],
        "params": _normalize_direction_params(custom_module_name, evaluation_params),
    }
    combo_count = len(expand_bias_specs(exploration_bias_spec))
    return UiModuleSelection(
        module_name=custom_module_name,
        display_label=custom_module_name,
        exploration_bias_spec=exploration_bias_spec,
        evaluation_bias_spec=evaluation_bias_spec,
        reports_dir=_reports_dir_for_module(custom_module_name, params),
        params_summary=f"{combo_count} custom combos from JSON parameter grid",
    )


def _stacked_sma_spec(
    timeframe: TimeFrame,
    combo: tuple[int, ...],
) -> dict[str, Any]:
    params = {
        f"period_{index + 1}": int(period)
        for index, period in enumerate(combo)
    }
    return {
        "module_name": "stacked_sma_long_only",
        "timeframes": [timeframe],
        "params": params,
    }


def _grid_value(values: list[int] | list[float]) -> int | float | list[int] | list[float]:
    return values if len(values) > 1 else values[0]


def _eval_value(values: list[int] | list[float]) -> int | float:
    return values[len(values) // 2]


def _parse_int_scalar(
    text: str,
    *,
    field_name: str,
    minimum: int,
) -> int:
    values = _parse_int_list(text, field_name=field_name, minimum=minimum)
    if len(values) != 1:
        raise ValueError(f"{field_name} must contain exactly one integer value.")
    return values[0]


def _parse_int_list(
    text: str,
    *,
    field_name: str,
    minimum: int = 1,
    allow_zero: bool = False,
) -> list[int]:
    values = _parse_csv_values(text, field_name=field_name)
    parsed = [int(value) for value in values]
    if not allow_zero and any(value < minimum for value in parsed):
        raise ValueError(f"{field_name} values must be >= {minimum}.")
    if allow_zero and any(value < 0 or (value != 0 and value < minimum) for value in parsed):
        raise ValueError(f"{field_name} values must be 0 or >= {minimum}.")
    return parsed


def _parse_float_list(text: str, *, field_name: str) -> list[float]:
    values = _parse_csv_values(text, field_name=field_name)
    return [float(value) for value in values]


def _parse_csv_values(text: str, *, field_name: str) -> list[str]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{field_name} must contain at least one value.")
    return list(dict.fromkeys(parts))


def _parse_json_object(text: str, *, field_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to a JSON object.")
    return {str(key): value for key, value in parsed.items()}


def _reports_dir_for_module(module_name: str, params: dict[str, str]) -> Path:
    fingerprint = hashlib.sha1(
        json.dumps(params, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:10]
    return _SIGNAL_RESULTS_DIR / f"{module_name}__{fingerprint}"


def _stacked_sma_defaults_from_config(config: ResearchConfig) -> dict[str, str]:
    bias_spec = config.bias_spec
    specs = bias_spec if isinstance(bias_spec, list) else expand_bias_specs(bias_spec)
    period_values = sorted(
        {
            int(value)
            for spec in specs
            for key, value in dict(spec.get("params", {})).items()
            if str(key).startswith("period_")
        }
    )
    first_spec = first_bias_spec(specs)
    params = dict(first_spec.get("params", {}))
    layer_count = len([key for key in params if str(key).startswith("period_")])
    return {
        "candidate_periods": ",".join(str(value) for value in period_values),
        "layer_count": str(layer_count),
    }


def _simple_defaults_from_spec(
    spec: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> dict[str, str]:
    params = dict(spec.get("params", {}))
    defaults = {
        key: _default_text_for_param(params.get(key))
        for key in keys
        if key in params
    }
    if "strategy_mode" in defaults:
        defaults["strategy_mode"] = _normalize_ui_strategy_mode(defaults["strategy_mode"])
    return defaults


def _default_text_for_param(value: object) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, PositionMode):
        return value.value
    return str(value)


def _normalize_ui_strategy_mode(value: object) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if not text or text == Direction.LONG_SHORT.value:
        return Direction.LONG.value
    if text not in {Direction.LONG.value, Direction.SHORT.value}:
        raise ValueError("strategy_mode must be 'long' or 'short'.")
    return text


def _normalize_direction_params(
    module_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(params)
    if module_name == "stacked_sma_long_only":
        normalized.pop("mode", None)
    if module_name == "cyclical_rsi_signal" and "strategy_mode" in normalized:
        raw_value = normalized["strategy_mode"]
        if isinstance(raw_value, list):
            normalized["strategy_mode"] = list(
                dict.fromkeys(_normalize_ui_strategy_mode(item) for item in raw_value)
            )
        else:
            normalized["strategy_mode"] = _normalize_ui_strategy_mode(raw_value)
    return normalized


def _catalog_option(
    *,
    module_name: str,
    fields: tuple[UiParameterField, ...],
) -> UiModuleOption:
    if module_name not in CANONICAL_MODULE_IMPORTS or module_name not in CANONICAL_MODULE_CLASSES:
        raise ValueError(f"Module {module_name!r} is not registered in nodes._taxonomy.")
    return UiModuleOption(
        value=module_name,
        label=module_name,
        description=CANONICAL_MODULE_CLASSES[module_name],
        fields=fields,
    )
