"""Canonical source priority helpers for data reconciliation policy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class CanonicalSourcePolicy:
    """Data source ranking used by ingestion/reconciliation jobs."""

    resolution_priority: dict[str, list[str]]
    instrument_class_priority: dict[str, dict[str, list[str]]]

    def select_priority(self, *, resolution: str, instrument_class: str) -> list[str]:
        cls_map = self.instrument_class_priority.get(instrument_class, {})
        if resolution in cls_map:
            return list(cls_map[resolution])
        return list(self.resolution_priority.get(resolution, []))


_DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "deployment"
    / "config"
    / "canonical_source_priority.json"
)


def load_canonical_source_policy(path: Path | None = None) -> CanonicalSourcePolicy:
    policy_path = path or _DEFAULT_POLICY_PATH
    raw = json.loads(policy_path.read_text(encoding="utf-8"))
    return CanonicalSourcePolicy(
        resolution_priority={
            str(k): [str(v) for v in values]
            for k, values in dict(raw.get("resolution_priority", {})).items()
        },
        instrument_class_priority={
            str(cls_name): {
                str(res): [str(v) for v in values]
                for res, values in dict(cls_rules).items()
            }
            for cls_name, cls_rules in dict(raw.get("instrument_class_priority", {})).items()
        },
    )
