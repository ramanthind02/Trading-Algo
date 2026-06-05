"""Serialize / deserialize the single winning param dict for walkforward outputs."""

from __future__ import annotations

import json
from enum import Enum
from typing import Mapping


def _json_safe_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(k): _json_safe_value(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            out = value.item()
        except Exception:
            out = None
        else:
            if isinstance(out, (str, int, float, bool)) or out is None:
                return out
    return str(value)


def serialize_selected_params(params: Mapping[str, object]) -> str:
    """JSON-encode one param dict for ``selection_summary_df.selected_params_json``."""
    return json.dumps(
        _json_safe_value(dict(params)),
        separators=(",", ":"),
        ensure_ascii=True,
    )


def decode_selected_params_list(payload: str) -> list[dict[str, object]]:
    """Decode summary JSON to a one-element list for portfolio / equity helpers."""
    try:
        raw = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return []
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [dict(raw)]
    return []
