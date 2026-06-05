"""Pivot-ready parameter sensitivity payloads for the research workspace UI."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import pandas as pd

from lib.core.repo_bootstrap import require_repo_root, resolve_repo_path

_FEATURE_REPO_ROOT = require_repo_root(Path(__file__))
from research.feature.config import ResearchConfig
from research.feature.filter_research_labels import build_filter_exploration_long_pairs
from research.feature.in_sample.data_loader import enrich_param_combo_with_module
from research.feature.in_sample.metric_helpers import SUPPORTED_PARAM_SENSITIVITY_METRICS
from research.feature.shared import FeatureResearchPhase
from research.feature.shared.visualization_paths import (
    canonical_in_sample_visualization_dir,
    walkforward_visualization_csv_dir,
)
from research.feature.ui.pivot_labels import (
    metric_display_label,
    param_field_display_label,
    param_field_group,
)
from research.feature.ui.workspace_manifest import validation_artifacts_current
from research.feature.visualization.parameter_sensitivity_explorer import (
    PreparedDataset,
    _format_param_value,
    _merge_parameter_dimensions,
    _parse_param_value,
    _value_options,
    _varying_param_columns,
    load_parameter_sensitivity_datasets,
)

_FILTER_GATE_PIVOT_FIELDS: tuple[str, ...] = (
    "filter_family",
    "gate_mode",
    "vol_max_rank",
    "gate_type",
    "filter_detail",
)
_FILTER_PIVOT_IDENTITY_COLUMNS: tuple[str, ...] = (
    "param_combo_label",
    "research_display_label",
    "feature_name",
)

_DEFAULT_METRIC = "t_stat"
_IDENTITY_COLUMNS = frozenset(
    {
        "param_combo_label",
        "feature_name",
        "param_sensitivity_by_ticker_key",
    }
)
_EXTRA_METRIC_COLUMNS = frozenset({"n_observations", "n_nonzero_signal"})


def visualization_dir_for_phase(
    config: ResearchConfig,
    phase: FeatureResearchPhase,
) -> Path | None:
    """Resolve the visualization CSV directory for one research phase."""

    if phase is FeatureResearchPhase.EXPLORATION:
        candidate = canonical_in_sample_visualization_dir()
    else:
        output_root = getattr(config, "output_root", None)
        if output_root is None:
            return None
        if not validation_artifacts_current(config):
            return None
        phase_subdir = "validation" if phase is FeatureResearchPhase.VALIDATION else "oos"
        candidate = resolve_repo_path(
            walkforward_visualization_csv_dir(Path(output_root), phase_subdir),
            repo_root=_FEATURE_REPO_ROOT,
        )
    return candidate if candidate.exists() else None


def pivot_payload_available(input_dir: Path) -> bool:
    """Return whether pivot explorer inputs exist (full signal grid and/or filter follow-up)."""

    filter_summary = input_dir / "filter_exploration_summary.csv"
    if filter_summary.is_file() and filter_summary.stat().st_size > 0:
        return True
    required = (
        input_dir / "param_sensitivity.csv",
        input_dir / "param_combo_long.csv",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _read_param_long_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    return frame if not frame.empty else None


def _filter_long_has_feature_params(long_df: pd.DataFrame) -> bool:
    if "param_key" not in long_df.columns:
        return False
    keys = long_df["param_key"].astype(str).unique()
    return any(
        key.startswith(("s_", "f_")) or key in {"filter_module", "signal_module"}
        for key in keys
    )


def _combo_label_from_metadata_path(metadata_path: Path) -> str | None:
    """Resolve ``filter_gate__combo_<hash>`` from nested signed_signal artifact paths."""
    for part in reversed(metadata_path.parent.parts):
        if "__combo_" in part:
            return part
    return None


def _rebuild_filter_param_long_from_combo_metadata(
    exploration_results_root: Path,
) -> pd.DataFrame | None:
    """Backfill ``s_*`` / ``f_*`` rows from per-combo ``metadata.json`` when CSVs are gate-only."""
    signed_signal_root = exploration_results_root / "signed_signal"
    if not signed_signal_root.is_dir():
        return None

    pairs: list[tuple[str, dict[str, Any]]] = []
    for metadata_path in signed_signal_root.glob("**/metadata.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        combo = payload.get("param_combo")
        if not isinstance(combo, dict):
            continue
        label = _combo_label_from_metadata_path(metadata_path)
        if label is None:
            continue
        module_token = label.split("__", 1)[0]
        enriched = enrich_param_combo_with_module(combo, module_token)
        pairs.extend(build_filter_exploration_long_pairs(label, enriched))

    if not pairs:
        return None
    return build_param_combo_long_table(pairs)


def _load_filter_exploration_param_long(input_dir: Path) -> pd.DataFrame | None:
    for stem in ("filter_exploration_long", "param_combo_long"):
        frame = _read_param_long_csv(input_dir / f"{stem}.csv")
        if frame is not None and _filter_long_has_feature_params(frame):
            return frame

    rebuilt = _rebuild_filter_param_long_from_combo_metadata(input_dir.parent)
    if rebuilt is not None and not rebuilt.empty:
        return rebuilt

    for stem in ("filter_exploration_long", "param_combo_long"):
        frame = _read_param_long_csv(input_dir / f"{stem}.csv")
        if frame is not None:
            return frame
    return None


def _gate_only_param_fields(frame: pd.DataFrame) -> list[str]:
    param_fields = [
        field
        for field in ("filter_family", "gate_mode", "vol_max_rank")
        if field in frame.columns and frame[field].nunique(dropna=False) > 1
    ]
    if len(param_fields) >= 2:
        return param_fields
    return [
        field
        for field in _FILTER_GATE_PIVOT_FIELDS
        if field in frame.columns and frame[field].nunique(dropna=False) > 1
    ][:2]


def _default_filter_pivot_axes(axis_params: tuple[str, ...]) -> tuple[str, str]:
    gate_fields = [field for field in axis_params if field in _FILTER_GATE_PIVOT_FIELDS]
    signal_fields = [field for field in axis_params if field.startswith("s_")]
    filter_param_fields = [field for field in axis_params if field.startswith("f_")]

    if "filter_family" in gate_fields and signal_fields:
        return "filter_family", signal_fields[0]

    row_field = signal_fields[0] if signal_fields else (
        gate_fields[0] if gate_fields else axis_params[0]
    )
    col_candidates = [
        field
        for field in (*gate_fields, *signal_fields, *filter_param_fields, *axis_params)
        if field != row_field
    ]
    col_field = col_candidates[0] if col_candidates else ""
    return row_field, col_field


def _build_filter_exploration_dataset(
    input_dir: Path,
    metric_columns: list[str],
) -> dict[str, object] | None:
    summary_path = input_dir / "filter_exploration_summary.csv"
    if not summary_path.exists():
        return None
    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return None
    if summary.empty:
        return None

    long_df = _load_filter_exploration_param_long(input_dir)
    pivot_dims: tuple[str, ...]
    param_fields: list[str]
    value_options: dict[str, tuple[str, ...] | list[str]]
    export_frame: pd.DataFrame

    if long_df is not None:
        summary_semantics = set(_FILTER_GATE_PIVOT_FIELDS) & set(summary.columns)
        long_for_merge = long_df
        if summary_semantics and "param_key" in long_df.columns:
            long_for_merge = long_df[
                ~long_df["param_key"].astype(str).isin(summary_semantics)
            ]
            if long_for_merge.empty:
                long_for_merge = long_df
        merged, ordered_params = _merge_parameter_dimensions(summary, long_for_merge)
        gate_dims = tuple(
            field for field in _FILTER_GATE_PIVOT_FIELDS if field in summary.columns
        )
        ordered_params = tuple(dict.fromkeys((*gate_dims, *ordered_params)))
        axis_params = _varying_param_columns(merged, ordered_params)
        if len(axis_params) >= 2:
            pivot_dims = tuple(
                dict.fromkeys(
                    (
                        *_FILTER_PIVOT_IDENTITY_COLUMNS,
                        *_FILTER_GATE_PIVOT_FIELDS,
                        *axis_params,
                    )
                )
            )
            export_columns = [
                column
                for column in (*pivot_dims, *metric_columns)
                if column in merged.columns
            ]
            export_frame = _format_dimension_columns(merged[export_columns], pivot_dims)
            param_fields = list(axis_params)
            value_options = {
                key: list(values)
                for key, values in _value_options(merged, axis_params).items()
            }
        else:
            long_df = None

    if long_df is None:
        pivot_dims = _FILTER_GATE_PIVOT_FIELDS
        export_columns = [
            column
            for column in (
                *_FILTER_PIVOT_IDENTITY_COLUMNS,
                *pivot_dims,
                *metric_columns,
            )
            if column in summary.columns
        ]
        export_frame = _format_dimension_columns(summary[export_columns], pivot_dims)
        param_fields = _gate_only_param_fields(summary)
        if len(param_fields) < 2:
            return None
        value_options = {
            field: tuple(sorted(summary[field].astype(str).unique(), key=str))
            for field in param_fields
        }

    records = _json_records(export_frame, metric_columns)
    return {
        "id": "filter_comparison",
        "label": "Gate + signal (filter exploration)",
        "param_fields": param_fields,
        "filter_fields": [],
        "value_options": value_options,
        "records": records,
    }


def build_parameter_sensitivity_pivot_payload(input_dir: Path) -> dict[str, object] | None:
    """Build a JSON-serializable pivot payload from exploration visualization CSVs."""

    if not pivot_payload_available(input_dir):
        return None

    datasets = load_parameter_sensitivity_datasets(input_dir)
    metric_columns = _shared_metric_columns(datasets) if datasets else []

    filter_dataset = _build_filter_exploration_dataset(input_dir, metric_columns)
    dataset_payloads: list[dict[str, object]] = []
    if filter_dataset is not None:
        dataset_payloads.append(filter_dataset)
    dataset_payloads.extend(
        _dataset_payload(dataset, metric_columns) for dataset in datasets
    )
    if not dataset_payloads:
        return None

    if not metric_columns and filter_dataset is not None:
        metric_columns = [
            column
            for column in ("sharpe", "t_stat", "sortino", "n_nonzero_signal", "trade_reduction_pct")
            if column in filter_dataset["records"][0]
        ]

    default_metric = "sharpe" if "sharpe" in metric_columns else (
        _DEFAULT_METRIC if _DEFAULT_METRIC in metric_columns else metric_columns[0]
    )

    scale: dict[str, float | None] = {
        "vmin": None,
        "vmax": None,
        "observed_min": None,
        "observed_max": None,
    }
    if datasets:
        scale = _global_metric_scale(datasets, default_metric)

    payload: dict[str, object] = {
        "default_metric": default_metric,
        "metrics": metric_columns,
        "metric_labels": {metric: metric_display_label(metric) for metric in metric_columns},
        "scale": scale,
        "datasets": dataset_payloads,
    }
    all_param_fields = {
        field
        for dataset in dataset_payloads
        for field in dataset["param_fields"]
    }
    all_filter_fields = {
        field
        for dataset in dataset_payloads
        for field in dataset.get("filter_fields", [])
    }
    label_fields = sorted(all_param_fields | all_filter_fields)
    payload["field_labels"] = {
        field: param_field_display_label(field) for field in label_fields
    }
    payload["field_groups"] = {
        field: param_field_group(field) for field in label_fields
    }
    if filter_dataset is not None:
        axis_params = tuple(filter_dataset["param_fields"])
        default_row, default_col = _default_filter_pivot_axes(axis_params)
        payload["exploration_mode"] = "filter_comparison"
        payload["default_dataset_id"] = "filter_comparison"
        payload["default_row_field"] = default_row
        payload["default_col_field"] = default_col
        payload["pivot_hint"] = (
            "Compare how a performance metric changes across two parameter dimensions. "
            "Each cell averages the metric for every combo with that row/column pair. "
            "For filter exploration, start with Filter family on rows and a signal parameter "
            "(for example entry lookback) on columns."
        )
    return payload


def pivot_explorer_metadata_for_phase(
    config: ResearchConfig,
    phase: FeatureResearchPhase,
) -> dict[str, object]:
    """Lightweight workspace hint for whether the pivot explorer should render."""

    input_dir = visualization_dir_for_phase(config, phase)
    if input_dir is None:
        return {"available": False, "phase": phase.value}
    return {
        "available": pivot_payload_available(input_dir),
        "phase": phase.value,
    }


def load_parameter_sensitivity_pivot_payload_for_phase(
    config: ResearchConfig,
    phase: FeatureResearchPhase,
) -> dict[str, object] | None:
    """Load pivot payload for one phase, or ``None`` when inputs are missing."""

    input_dir = visualization_dir_for_phase(config, phase)
    if input_dir is None:
        return None
    return build_parameter_sensitivity_pivot_payload(input_dir)


def _shared_metric_columns(datasets: tuple[PreparedDataset, ...]) -> list[str]:
    discovered = {
        column
        for dataset in datasets
        for column in dataset.frame.columns
        if column in SUPPORTED_PARAM_SENSITIVITY_METRICS or column in _EXTRA_METRIC_COLUMNS
    }
    ordered = [metric for metric in sorted(SUPPORTED_PARAM_SENSITIVITY_METRICS) if metric in discovered]
    extras = sorted(column for column in discovered if column not in ordered)
    return [*ordered, *extras]


def _dataset_payload(
    dataset: PreparedDataset,
    metric_columns: list[str],
) -> dict[str, object]:
    dimension_columns = tuple(
        [*dataset.filter_dimensions, *dataset.axis_params, "param_combo_label"]
    )
    frame = _format_dimension_columns(dataset.frame, dimension_columns)
    export_columns = [
        column
        for column in (*dimension_columns, *metric_columns)
        if column in frame.columns
    ]
    records = _json_records(frame[export_columns], metric_columns)
    return {
        "id": dataset.name,
        "label": dataset.title,
        "param_fields": list(dataset.axis_params),
        "filter_fields": list(dataset.filter_dimensions),
        "value_options": {
            key: list(values) for key, values in dataset.value_options.items()
        },
        "records": records,
    }


def _format_dimension_columns(
    frame: pd.DataFrame,
    dimension_columns: tuple[str, ...],
) -> pd.DataFrame:
    formatted = frame.copy()
    for column in dimension_columns:
        if column not in formatted.columns:
            continue
        formatted[column] = formatted[column].map(
            lambda value: (
                None
                if pd.isna(value)
                else _format_param_value(_parse_param_value(value))
            )
        )
    return formatted


def _json_records(frame: pd.DataFrame, metric_columns: list[str]) -> list[dict[str, object]]:
    rows = frame.to_dict(orient="records")
    return [_json_row(row, metric_columns) for row in rows]


def _json_row(row: dict[str, object], metric_columns: list[str]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in row.items():
        if key in metric_columns:
            payload[key] = _json_number(value)
            continue
        if key in _IDENTITY_COLUMNS or pd.isna(value):
            payload[key] = None if pd.isna(value) else str(value)
            continue
        payload[key] = str(value)
    return payload


def _json_number(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if pd.isna(value):
        return None
    return float(value)


def _global_metric_scale(
    datasets: tuple[PreparedDataset, ...],
    metric: str,
) -> dict[str, float | None]:
    values = [
        pd.to_numeric(dataset.frame[metric], errors="coerce")
        for dataset in datasets
        if metric in dataset.frame.columns
    ]
    if not values:
        return {"vmin": None, "vmax": None, "observed_min": None, "observed_max": None}
    series = pd.concat(values, ignore_index=True).dropna()
    if series.empty:
        return {"vmin": None, "vmax": None, "observed_min": None, "observed_max": None}
    observed_min = float(series.min())
    observed_max = float(series.max())
    return {
        "observed_min": observed_min,
        "observed_max": observed_max,
        "vmin": observed_min,
        "vmax": observed_max if not math.isclose(observed_min, observed_max) else observed_max + 1e-9,
    }
