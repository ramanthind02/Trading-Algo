"""Nautilus-aligned BarType and BarSpecification.

String form matches NautilusTrader exactly:
  {instrument_id}-{step}-{aggregation}-{price_type}-{source}
e.g.  AAPL.XNAS-1-DAY-LAST-EXTERNAL
      ES.XCME-1-DAY-LAST-EXTERNAL

Composite (bar-to-bar) form with ``@`` is also parsed/rendered:
  ES.XCME-1-WEEK-LAST-INTERNAL@1-DAY-EXTERNAL

This is the canonical key for stored bar series in the catalog. The repo's
internal ``TimeFrame`` enum maps to/from BarSpecification via ``from_timeframe``
/ ``to_timeframe`` so existing pipeline code keeps working during migration.
"""
from __future__ import annotations

from dataclasses import dataclass

from .enums import AggregationSource, BarAggregation, PriceType
from .identifiers import InstrumentId

# repo TimeFrame.name -> (step, BarAggregation)
_TIMEFRAME_TO_SPEC: dict[str, tuple[int, BarAggregation]] = {
    "H1": (1, BarAggregation.HOUR),
    "H4": (4, BarAggregation.HOUR),
    "D": (1, BarAggregation.DAY),
    "W": (1, BarAggregation.WEEK),
    "M": (1, BarAggregation.MONTH),
}
_SPEC_TO_TIMEFRAME: dict[tuple[int, BarAggregation], str] = {
    v: k for k, v in _TIMEFRAME_TO_SPEC.items()
}


@dataclass(frozen=True, slots=True)
class BarSpecification:
    step: int
    aggregation: BarAggregation
    price_type: PriceType

    def __str__(self) -> str:
        return f"{self.step}-{self.aggregation.value}-{self.price_type.value}"

    @classmethod
    def from_timeframe(cls, timeframe_name: str, price_type: PriceType = PriceType.LAST) -> "BarSpecification":
        if timeframe_name not in _TIMEFRAME_TO_SPEC:
            raise ValueError(f"Unknown TimeFrame name: {timeframe_name!r}")
        step, agg = _TIMEFRAME_TO_SPEC[timeframe_name]
        return cls(step, agg, price_type)

    def to_timeframe(self) -> str:
        key = (self.step, self.aggregation)
        if key not in _SPEC_TO_TIMEFRAME:
            raise ValueError(f"BarSpecification {self} has no TimeFrame mapping")
        return _SPEC_TO_TIMEFRAME[key]


@dataclass(frozen=True, slots=True)
class BarType:
    instrument_id: InstrumentId
    spec: BarSpecification
    aggregation_source: AggregationSource = AggregationSource.EXTERNAL
    composite: "BarType | None" = None  # source bar type for bar-to-bar (@ part)

    def __str__(self) -> str:
        base = f"{self.instrument_id}-{self.spec}-{self.aggregation_source.value}"
        if self.composite is not None:
            c = self.composite
            base += f"@{c.spec.step}-{c.spec.aggregation.value}-{c.aggregation_source.value}"
        return base

    @property
    def value(self) -> str:
        return str(self)

    @classmethod
    def from_str(cls, value: str) -> "BarType":
        target_part, _, source_part = value.partition("@")
        bar_type = cls._parse_standard(target_part)
        if source_part:
            # source side: {step}-{agg}-{source}, instrument + price_type inherited
            step_s, agg_s, src_s = source_part.split("-")
            src_spec = BarSpecification(int(step_s), BarAggregation(agg_s), bar_type.spec.price_type)
            composite = cls(bar_type.instrument_id, src_spec, AggregationSource(src_s))
            return cls(bar_type.instrument_id, bar_type.spec, bar_type.aggregation_source, composite)
        return bar_type

    @staticmethod
    def _parse_standard(value: str) -> "BarType":
        # {instrument_id}-{step}-{aggregation}-{price_type}-{source}
        # instrument_id contains exactly one dot and may contain no '-',
        # so split the trailing 4 fields off the right.
        head, step_s, agg_s, price_s, src_s = value.rsplit("-", 4)
        return BarType(
            instrument_id=InstrumentId.from_str(head),
            spec=BarSpecification(int(step_s), BarAggregation(agg_s), PriceType(price_s)),
            aggregation_source=AggregationSource(src_s),
        )
