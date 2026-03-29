from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from utils.core.enums import Direction, DirectionInput, Ticker, TimeFrame, coerce_direction


MIGRATION_ERROR_MESSAGE = (
    "Legacy fitted feature artifacts are unsupported after the domain-discrete cutover. "
    "Regenerate the feature/control file as model_type='domain_discrete' with a frozen bias_node_spec."
)


class DomainDiscreteMigrationError(ValueError):
    """Raised when legacy fitted feature payloads are encountered."""


class TickerScopeKind(Enum):
    SINGLE = "single"
    GROUP = "group"


def raise_legacy_feature_artifact(details: str) -> None:
    raise DomainDiscreteMigrationError(f"{MIGRATION_ERROR_MESSAGE} {details}")


def _coerce_timeframe(raw: object) -> TimeFrame:
    if isinstance(raw, TimeFrame):
        return raw
    if isinstance(raw, str):
        return TimeFrame[raw]
    raise TypeError(f"timeframe must be TimeFrame or str, got {type(raw).__name__}")


def _coerce_ticker(raw: object) -> Ticker:
    if isinstance(raw, Ticker):
        return raw
    if isinstance(raw, str):
        return Ticker[raw]
    raise TypeError(f"ticker must be Ticker or str, got {type(raw).__name__}")


def _coerce_bin_indexes(raw: Sequence[object], *, field_name: str, n_bins: int) -> tuple[int, ...]:
    bins = tuple(int(item) for item in raw)
    if len(set(bins)) != len(bins):
        raise ValueError(f"{field_name} must not contain duplicates")
    if any(bin_idx < 0 or bin_idx >= n_bins for bin_idx in bins):
        raise ValueError(f"{field_name} must be within [0, {n_bins - 1}]")
    return bins


@dataclass(frozen=True)
class SourceBiasNodeSpec:
    module_name: str
    timeframes: tuple[TimeFrame, ...]
    params: dict[str, Any] = field(default_factory=dict)
    filters: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.module_name.strip():
            raise ValueError("source_bias_node_spec.module_name must be non-empty")
        if not self.timeframes:
            raise ValueError("source_bias_node_spec.timeframes must be non-empty")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SourceBiasNodeSpec:
        return cls(
            module_name=str(payload["module_name"]).strip(),
            timeframes=tuple(_coerce_timeframe(tf) for tf in payload["timeframes"]),
            params=dict(payload.get("params", {})),
            filters=tuple(dict(spec) for spec in payload.get("filters", ())),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "timeframes": [tf.name for tf in self.timeframes],
            "params": dict(self.params),
            "filters": [dict(spec) for spec in self.filters],
        }


@dataclass(frozen=True)
class TickerScope:
    tickers: tuple[Ticker, ...]
    scope_name: str | None = None

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("ticker_scope must declare at least one ticker")

    @property
    def kind(self) -> TickerScopeKind:
        return TickerScopeKind.SINGLE if len(self.tickers) == 1 else TickerScopeKind.GROUP

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TickerScope:
        tickers = tuple(_coerce_ticker(raw) for raw in payload.get("tickers", ()))
        scope_name = payload.get("scope_name")
        normalized_scope_name = str(scope_name).strip() if scope_name is not None else None
        return cls(tickers=tickers, scope_name=normalized_scope_name or None)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "tickers": [ticker.name for ticker in self.tickers],
            "scope_name": self.scope_name,
        }

    def allows(self, ticker: Ticker) -> bool:
        return ticker in self.tickers


@dataclass(frozen=True)
class DomainDiscreteSpec:
    source_bias_node_spec: SourceBiasNodeSpec
    ticker_scope: TickerScope
    edges: tuple[float, ...]
    n_bins: int
    long_bins: tuple[int, ...]
    short_bins: tuple[int, ...]
    direction: Direction
    spec_version: str

    def __post_init__(self) -> None:
        if not self.spec_version.strip():
            raise ValueError("spec_version must be non-empty")
        if len(self.edges) + 1 != self.n_bins:
            raise ValueError("n_bins must equal len(edges) + 1")
        if tuple(sorted(self.edges)) != self.edges:
            raise ValueError("edges must be strictly monotone")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("edges must be strictly monotone")
        if set(self.long_bins) & set(self.short_bins):
            raise ValueError("long_bins and short_bins must be disjoint")
        if self.direction is Direction.LONG:
            if not self.long_bins or self.short_bins:
                raise ValueError("direction='long' requires non-empty long_bins and empty short_bins")
        elif self.direction is Direction.SHORT:
            if not self.short_bins or self.long_bins:
                raise ValueError("direction='short' requires non-empty short_bins and empty long_bins")
        elif self.direction is Direction.LONG_SHORT:
            if not self.long_bins or not self.short_bins:
                raise ValueError(
                    "direction='long_short' requires both long_bins and short_bins"
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DomainDiscreteSpec:
        source_bias_node_spec = SourceBiasNodeSpec.from_mapping(payload["source_bias_node_spec"])
        ticker_scope = TickerScope.from_mapping(payload["ticker_scope"])
        n_bins = int(payload["n_bins"])
        edges = tuple(float(edge) for edge in payload["edges"])
        return cls(
            source_bias_node_spec=source_bias_node_spec,
            ticker_scope=ticker_scope,
            edges=edges,
            n_bins=n_bins,
            long_bins=_coerce_bin_indexes(payload.get("long_bins", ()), field_name="long_bins", n_bins=n_bins),
            short_bins=_coerce_bin_indexes(payload.get("short_bins", ()), field_name="short_bins", n_bins=n_bins),
            direction=coerce_direction(payload["direction"], field_name="direction"),
            spec_version=str(payload["spec_version"]).strip(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_bias_node_spec": self.source_bias_node_spec.to_mapping(),
            "ticker_scope": self.ticker_scope.to_mapping(),
            "edges": list(self.edges),
            "n_bins": self.n_bins,
            "long_bins": list(self.long_bins),
            "short_bins": list(self.short_bins),
            "direction": self.direction.value,
            "spec_version": self.spec_version,
        }


def load_domain_discrete_spec(payload: Mapping[str, Any] | DomainDiscreteSpec) -> DomainDiscreteSpec:
    if isinstance(payload, DomainDiscreteSpec):
        return payload
    return DomainDiscreteSpec.from_mapping(payload)


def build_domain_discrete_bias_node_spec(
    spec: Mapping[str, Any] | DomainDiscreteSpec,
) -> dict[str, Any]:
    domain_spec = load_domain_discrete_spec(spec)
    source_spec = domain_spec.source_bias_node_spec
    return {
        "module_name": "domain_discrete",
        "timeframes": [tf.name for tf in source_spec.timeframes],
        "params": domain_spec.to_mapping(),
    }
