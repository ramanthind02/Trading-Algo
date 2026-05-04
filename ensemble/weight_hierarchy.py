"""Manual nested hierarchy for equal-split weight allocation in the weight layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Sequence, Tuple, Union

import pandas as pd

from ensemble.portfolio_impl.global_weight_layer_adapter import build_global_stream_id

# JSON / parse output: groups and leaves (stream id or matcher).


@dataclass(frozen=True)
class _ResolvedLeaf:
    stream_id: str


@dataclass(frozen=True)
class _Group:
    group_id: str
    children: Tuple[Union["_Group", _ResolvedLeaf], ...]


def _expect_mapping(raw: object, ctx: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{ctx} must be a JSON object, got {type(raw).__name__}")
    return raw


def _parse_leaf(raw: Mapping[str, object], ctx: str) -> _ResolvedLeaf:
    if raw.get("type") != "leaf":
        raise ValueError(f"{ctx}: leaf node must have \"type\": \"leaf\"")
    if "stream_id" in raw:
        sid = raw["stream_id"]
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError(f"{ctx}: stream_id must be a non-empty string")
        return _ResolvedLeaf(stream_id=sid.strip())
    required = ("ticker", "timeframe", "model_name")
    if all(k in raw for k in required):
        t, tf, m = raw["ticker"], raw["timeframe"], raw["model_name"]
        if not all(isinstance(x, str) and str(x).strip() for x in (t, tf, m)):
            raise ValueError(f"{ctx}: ticker, timeframe, model_name must be non-empty strings")
        return _ResolvedLeaf(
            stream_id=build_global_stream_id(str(t).strip(), str(tf).strip(), str(m).strip())
        )
    raise ValueError(
        f"{ctx}: leaf must specify \"stream_id\" or "
        f"all of {list(required)}"
    )


def _parse_group(raw: Mapping[str, object], ctx: str) -> _Group:
    if raw.get("type") != "group":
        raise ValueError(f'{ctx}: expected "type": "group"')
    gid = raw.get("id")
    if not isinstance(gid, str) or not gid.strip():
        raise ValueError(f'{ctx}: group must have non-empty string "id"')
    children_raw = raw.get("children")
    if not isinstance(children_raw, Sequence) or isinstance(children_raw, (str, bytes)):
        raise ValueError(f'{ctx}: group must have a "children" array')
    children: List[Union[_Group, _ResolvedLeaf]] = []
    for idx, child in enumerate(children_raw):
        cm = f"{ctx}.children[{idx}]"
        child_map = _expect_mapping(child, cm)
        ctype = child_map.get("type")
        if ctype == "group":
            children.append(_parse_group(child_map, cm))
        elif ctype == "leaf":
            children.append(_parse_leaf(child_map, cm))
        else:
            raise ValueError(f'{cm}: node must have "type" "group" or "leaf"')
    if not children:
        raise ValueError(f"{ctx}: group must have at least one child")
    return _Group(group_id=gid.strip(), children=tuple(children))


def parse_hierarchy_spec(raw: Mapping[str, object]) -> _Group:
    """Validate and parse a hierarchy JSON object into a root group."""
    return _parse_group(raw, "hierarchy")


def load_hierarchy_from_path(path: str | Path) -> _Group:
    """Load hierarchy JSON from disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"hierarchy file not found: {p}")
    with p.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_hierarchy_spec(_expect_mapping(payload, "hierarchy file root"))


def resolve_hierarchy_for_fit(
    *,
    hierarchy_spec: Mapping[str, object] | None,
    hierarchy_path: str | None,
) -> _Group:
    """Resolve config into a root group; ``hierarchy_spec`` takes precedence over path."""
    if hierarchy_spec is not None:
        return parse_hierarchy_spec(hierarchy_spec)
    if hierarchy_path is not None and str(hierarchy_path).strip():
        return load_hierarchy_from_path(hierarchy_path)
    raise ValueError("hierarchy_equal requires hierarchy_spec or hierarchy_path")


def _collect_declared_stream_ids(node: Union[_Group, _ResolvedLeaf]) -> FrozenSet[str]:
    if isinstance(node, _ResolvedLeaf):
        return frozenset({node.stream_id})
    return frozenset().union(
        *(_collect_declared_stream_ids(ch) for ch in node.children)
    )


def validate_strict_stream_coverage(
    declared: FrozenSet[str],
    available_models: FrozenSet[str],
) -> None:
    """Raise ``ValueError`` if declared leaves and fitted streams differ."""
    if declared == available_models:
        return
    missing = sorted(declared - available_models)
    extra = sorted(available_models - declared)
    parts: List[str] = []
    if missing:
        parts.append(f"hierarchy declares streams not in forecasts: {missing}")
    if extra:
        parts.append(f"forecasts contain streams not in hierarchy: {extra}")
    raise ValueError("; ".join(parts))


def _walk_assign(
    node: Union[_Group, _ResolvedLeaf],
    ancestor_ids: Tuple[str, ...],
    mass: float,
    weights: Dict[str, float],
    assignments: Dict[str, str],
) -> None:
    if isinstance(node, _ResolvedLeaf):
        path = "/".join(ancestor_ids + (node.stream_id,))
        weights[node.stream_id] = mass
        assignments[node.stream_id] = path
        return
    n_children = len(node.children)
    share = mass / float(n_children)
    my_prefix = ancestor_ids + (node.group_id,)
    for ch in node.children:
        _walk_assign(ch, my_prefix, share, weights, assignments)


def compute_equal_split_weights(
    root: _Group,
    available_models: Sequence[str],
) -> Tuple[pd.Series, Dict[str, str], Dict[str, float], Dict[str, Dict[str, float | int | None]]]:
    """
    Equal split among siblings at each level; strict coverage vs ``available_models``.

    Returns model weights (normalized), cluster_assignments (model -> path),
    cluster_weights (each path prefix -> total leaf weight under that prefix),
    cluster_metrics (per path: member_count leaf descendants).
    """
    avail = frozenset(str(m) for m in available_models)
    declared = _collect_declared_stream_ids(root)
    validate_strict_stream_coverage(declared, avail)

    leaf_weights: Dict[str, float] = {}
    assignments: Dict[str, str] = {}
    _walk_assign(root, (), 1.0, leaf_weights, assignments)

    series = pd.Series(
        {m: float(leaf_weights[m]) for m in sorted(leaf_weights)}, dtype=float
    )

    cluster_weights: Dict[str, float] = {}
    for sid, path in assignments.items():
        mass = float(leaf_weights[sid])
        parts = path.split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            cluster_weights[prefix] = cluster_weights.get(prefix, 0.0) + mass

    cluster_metrics: Dict[str, Dict[str, float | int | None]] = {}
    for path in cluster_weights:
        cluster_metrics[path] = {
            "avg_positive_corr": 0.0,
            "ulcer_index": None,
            "score": None,
            "member_count": int(_count_leaves_under_prefix(path, assignments)),
        }
    return series, assignments, cluster_weights, cluster_metrics


def _count_leaves_under_prefix(prefix: str, assignments: Dict[str, str]) -> int:
    return sum(
        1
        for p in assignments.values()
        if p == prefix or p.startswith(prefix + "/")
    )


__all__ = [
    "compute_equal_split_weights",
    "load_hierarchy_from_path",
    "parse_hierarchy_spec",
    "resolve_hierarchy_for_fit",
    "validate_strict_stream_coverage",
]
