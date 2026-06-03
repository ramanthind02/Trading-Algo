"""Build portfolio-research weight-layer JSON for the workspace UI."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_research.config import (
    PortfolioResearchConfig,
    describe_weight_layer_policy,
    rebuild_weight_layer_kwargs,
)
from portfolio_research.weight_layer_export import _GLOBAL_SYNTHETIC_TICKER

_GLOBAL_TICKER = _GLOBAL_SYNTHETIC_TICKER


def _global_stream_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    mask = df["weight_layer_ticker"].astype(str) == _GLOBAL_TICKER
    return df.loc[mask].copy()


def _hierarchy_segments_from_cluster_id(
    cluster_id: object,
) -> tuple[str | None, str | None, str | None]:
    if cluster_id is None or (isinstance(cluster_id, float) and pd.isna(cluster_id)):
        return None, None, None
    path = str(cluster_id).strip()
    if not path:
        return None, None, None
    parts = path.split("/")
    if parts and parts[0] == "root":
        parts = parts[1:]
    if len(parts) < 2:
        return None, None, path
    return parts[0], parts[1], path


def _parse_stream_ticker(stream_id: str) -> str | None:
    parts = str(stream_id).strip().split("::", 2)
    return parts[0].strip().upper() if parts and parts[0].strip() else None


def _asset_class_budgets(streams: pd.DataFrame) -> dict[str, float]:
    budgets: dict[str, float] = {}
    for _, row in streams.iterrows():
        asset, _, _ = _hierarchy_segments_from_cluster_id(row.get("cluster_id"))
        if asset is None:
            continue
        budgets[asset] = budgets.get(asset, 0.0) + float(row["stream_weight"])
    return dict(sorted(budgets.items()))


def build_weight_layer_member_records(
    weight_layer_df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Long-format stream weights for a single fitted portfolio snapshot."""
    streams = _global_stream_rows(weight_layer_df)
    if streams.empty:
        return []

    rows: list[dict[str, object]] = []
    for _, row in streams.sort_values("stream_weight", ascending=False).iterrows():
        stream_id = str(row["stream_or_model_id"])
        weight = max(0.0, float(row["stream_weight"]))
        asset_class, style_group, hierarchy_path = _hierarchy_segments_from_cluster_id(
            row.get("cluster_id")
        )
        model_name = row.get("stream_source_model_name") or stream_id
        rows.append(
            {
                "stream_id": stream_id,
                "model_name": str(model_name),
                "hierarchy_path": hierarchy_path,
                "asset_class": asset_class,
                "style_group": style_group,
                "weight_without": weight,
                "weight_with": weight,
                "weight_delta": 0.0,
                "is_new_stream": False,
                "is_candidate_stream": False,
                "candidate_key": "",
            }
        )
    return rows


def build_weight_layer_tree_from_members(
    member_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Nest stream rows into asset → style → stream for UI rendering."""
    assets: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in member_records:
        asset = str(row.get("asset_class") or "unknown")
        style = str(row.get("style_group") or "unknown")
        assets.setdefault(asset, {}).setdefault(style, []).append(
            {
                "type": "stream",
                "stream_id": row.get("stream_id"),
                "model_name": row.get("model_name"),
                "ticker": _parse_stream_ticker(str(row.get("stream_id") or "")),
                "hierarchy_path": row.get("hierarchy_path"),
                "weight_without": float(row.get("weight_without") or 0.0),
                "weight_with": float(row.get("weight_with") or 0.0),
                "weight_delta": float(row.get("weight_delta") or 0.0),
                "is_candidate_stream": bool(row.get("is_candidate_stream")),
                "is_new_stream": bool(row.get("is_new_stream")),
            }
        )

    tree: list[dict[str, object]] = []
    for asset_id in sorted(assets.keys()):
        style_map = assets[asset_id]
        style_nodes: list[dict[str, object]] = []
        for style_id in sorted(style_map.keys()):
            stream_rows = sorted(
                style_map[style_id],
                key=lambda item: float(item.get("weight_with") or 0.0),
                reverse=True,
            )
            style_nodes.append(
                {
                    "type": "style",
                    "id": style_id,
                    "weight_without": sum(float(s["weight_without"]) for s in stream_rows),
                    "weight_with": sum(float(s["weight_with"]) for s in stream_rows),
                    "weight_delta": 0.0,
                    "stream_count": len(stream_rows),
                    "children": stream_rows,
                }
            )
        tree.append(
            {
                "type": "asset",
                "id": asset_id,
                "weight_without": sum(float(s["weight_without"]) for s in style_nodes),
                "weight_with": sum(float(s["weight_with"]) for s in style_nodes),
                "weight_delta": 0.0,
                "stream_count": sum(int(s["stream_count"]) for s in style_nodes),
                "children": style_nodes,
            }
        )
    return tree


def _hierarchy_outline_from_spec(spec: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def _walk(node: Mapping[str, object], path_parts: tuple[str, ...]) -> None:
        node_type = str(node.get("type", ""))
        if node_type == "leaf":
            sid = str(node.get("stream_id", "")).strip()
            rows.append(
                {
                    "path": "/".join(path_parts),
                    "kind": "leaf",
                    "stream_id": sid,
                    "ticker": _parse_stream_ticker(sid),
                }
            )
            return
        if node_type != "group":
            return
        group_id = str(node.get("id", "")).strip()
        next_path = path_parts + (group_id,) if group_id else path_parts
        children = node.get("children")
        if not isinstance(children, list):
            return
        leaf_count = sum(
            1
            for child in children
            if isinstance(child, Mapping) and str(child.get("type")) == "leaf"
        )
        group_count = sum(
            1
            for child in children
            if isinstance(child, Mapping) and str(child.get("type")) == "group"
        )
        rows.append(
            {
                "path": "/".join(next_path),
                "kind": "group",
                "child_leaves": leaf_count,
                "child_groups": group_count,
            }
        )
        for child in children:
            if isinstance(child, Mapping):
                _walk(child, next_path)

    _walk(spec, ())
    return rows


def _fdm_from_diagnostics(weight_layer_df: pd.DataFrame) -> float | None:
    streams = _global_stream_rows(weight_layer_df)
    if streams.empty or "fdm_multiplier" not in streams.columns:
        return None
    values = streams["fdm_multiplier"].dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def build_portfolio_weight_layer_report(
    weight_layer_df: pd.DataFrame,
    *,
    config: PortfolioResearchConfig,
    phase: str,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    predict_start: pd.Timestamp,
    predict_end: pd.Timestamp,
) -> dict[str, object]:
    """Serializable weight-layer snapshot for portfolio research UI."""
    member_records = build_weight_layer_member_records(weight_layer_df)
    streams = _global_stream_rows(weight_layer_df)
    wl_kwargs = rebuild_weight_layer_kwargs(config)
    hierarchy_spec = wl_kwargs.get("hierarchy_spec")
    budgets = _asset_class_budgets(streams)
    fdm = _fdm_from_diagnostics(weight_layer_df)

    detail: dict[str, object] = {
        "method": config.weight_layer_method,
        "weight_layer_policy": describe_weight_layer_policy(wl_kwargs),
        "sr_adjustment": bool(wl_kwargs.get("sr_adjustment")),
        "sr_tilt_max_depth": wl_kwargs.get("sr_tilt_max_depth"),
        "within_group_method": wl_kwargs.get("within_group_method", "equal"),
        "fdm_max": wl_kwargs.get("fdm_max", 2.0),
        "fdm_without": fdm,
        "fdm_with": fdm,
        "stream_count_without": len(streams),
        "stream_count_with": len(streams),
        "ensemble_count_without": len(config.ensemble_dirs),
        "ensemble_count_with": len(config.ensemble_dirs),
        "asset_budgets_without": budgets,
        "asset_budgets_with": budgets,
        "tree": build_weight_layer_tree_from_members(member_records),
        "snapshot_mode": True,
    }
    if config.weight_layer_method == "hierarchy_equal":
        detail["hierarchy_mode"] = "asset_first"
        if isinstance(hierarchy_spec, Mapping):
            detail["hierarchy_spec_with"] = dict(hierarchy_spec)
            detail["hierarchy_outline_with"] = _hierarchy_outline_from_spec(hierarchy_spec)

    return {
        "meta": {
            "phase": phase,
            "fit_start": fit_start.date().isoformat(),
            "fit_end": fit_end.date().isoformat(),
            "predict_start": predict_start.date().isoformat(),
            "predict_end": predict_end.date().isoformat(),
            "ensemble_count": len(config.ensemble_dirs),
            "stream_count": len(streams),
            "snapshot_mode": True,
        },
        "context": {
            "weight_layer_method": config.weight_layer_method,
            "weight_layer_policy": describe_weight_layer_policy(wl_kwargs),
            "weight_layer_hierarchy_mode": "asset_first"
            if config.weight_layer_method == "hierarchy_equal"
            else None,
            "weight_layer_members": member_records,
            "weight_layer_detail": detail,
            "asset_class_budgets": budgets,
        },
    }


def write_portfolio_weight_layer_report(
    weight_layer_df: pd.DataFrame,
    path: Path,
    *,
    config: PortfolioResearchConfig,
    phase: str,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    predict_start: pd.Timestamp,
    predict_end: pd.Timestamp,
) -> Path:
    """Write weight-layer JSON report for workspace UI."""
    payload = build_portfolio_weight_layer_report(
        weight_layer_df,
        config=config,
        phase=phase,
        fit_start=fit_start,
        fit_end=fit_end,
        predict_start=predict_start,
        predict_end=predict_end,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
