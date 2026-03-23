from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional

from utils.core.enums import Ticker, TimeFrame


class ArtifactScope(str, Enum):
    LIVE = "live"
    RESEARCH = "research"


class ArtifactLifecycleState(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    REBUILDING = "rebuilding"
    FAILED = "failed"


class LookupMode(str, Enum):
    EXACT = "exact"
    AS_OF = "as_of"


def _normalize_params_signature(params: Mapping[str, Any]) -> str:
    if not params:
        return "{}"
    return json.dumps(params, sort_keys=True, default=str)


@dataclass(frozen=True)
class CoverageWindow:
    start: Optional[datetime]
    end: Optional[datetime]


@dataclass(frozen=True)
class CacheRequest:
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    exact_dt: Optional[datetime] = None
    as_of_dt: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.exact_dt is not None and self.as_of_dt is not None:
            raise ValueError("CacheRequest can only specify one of exact_dt or as_of_dt.")


@dataclass(frozen=True)
class ArtifactDescriptor:
    family: str
    ticker: Optional[Ticker] = None
    timeframe: Optional[TimeFrame] = None
    module_name: Optional[str] = None
    params: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    scope: ArtifactScope = ArtifactScope.LIVE
    artifact_name: Optional[str] = None
    name: Optional[str] = None
    params_signature: str = field(init=False)

    def __post_init__(self) -> None:
        if self.artifact_name is None and self.name is not None:
            object.__setattr__(self, "artifact_name", self.name)
        if self.name is None and self.artifact_name is not None:
            object.__setattr__(self, "name", self.artifact_name)
        object.__setattr__(self, "params_signature", _normalize_params_signature(self.params))

    def cache_key(self) -> tuple[str, str, str, str, str]:
        ticker = self.ticker.name if self.ticker is not None else "*"
        timeframe = self.timeframe.name if self.timeframe is not None else "*"
        module_name = self.module_name or "*"
        artifact_name = self.artifact_name or "*"
        return (self.family, module_name, ticker, timeframe, artifact_name)

    def with_scope(self, scope: ArtifactScope) -> "ArtifactDescriptor":
        return ArtifactDescriptor(
            family=self.family,
            ticker=self.ticker,
            timeframe=self.timeframe,
            module_name=self.module_name,
            params=dict(self.params),
            scope=scope,
            artifact_name=self.artifact_name,
        )


@dataclass(frozen=True)
class ArtifactRecord:
    descriptor: ArtifactDescriptor
    coverage: CoverageWindow
    lifecycle_state: ArtifactLifecycleState
    revision: int
    source_revision: int = 0
    depends_on: tuple[tuple[str, str], ...] = ()
