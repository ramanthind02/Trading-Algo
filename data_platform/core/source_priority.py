"""Config-driven source priority for multi-source reconciliation.

Resolves which source (norgate / mt5 / derived_from_daily) wins for a given
(instrument_class, resolution), evaluated per-instrument at ingestion.

See docs/library/Data/multi_source_update_architecture.md §3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .enums import BarAggregation, InstrumentClass

# YAML uses repo timeframe short codes; map them to BarAggregation members.
_RESOLUTION_CODE_TO_AGG: dict[str, BarAggregation] = {
    "D": BarAggregation.DAY,
    "W": BarAggregation.WEEK,
    "M": BarAggregation.MONTH,
    "M1": BarAggregation.MINUTE,
}


@dataclass(frozen=True)
class SourceRule:
    source: str       # "norgate" | "mt5" | "derived_from_daily"
    condition: str    # "always" | "norgate_active == True" | "norgate_active == False"


@dataclass(frozen=True)
class PriorityRule:
    instrument_class: InstrumentClass
    resolutions: tuple[BarAggregation, ...]   # empty tuple = matches all resolutions
    priority: tuple[SourceRule, ...]


def _eval_condition(condition: str, ctx: dict[str, bool]) -> bool:
    cond = condition.strip()
    if cond == "always":
        return True
    if cond == "norgate_active == True":
        return ctx.get("norgate_active", False) is True
    if cond == "norgate_active == False":
        return ctx.get("norgate_active", False) is False
    raise ValueError(f"Unsupported source-priority condition: {condition!r}")


@dataclass
class SourcePriorityConfig:
    rules: list[PriorityRule] = field(default_factory=list)
    norgate_active: bool = True

    def active_source(
        self,
        instrument_class: InstrumentClass,
        resolution: BarAggregation,
    ) -> str | None:
        """First source whose condition holds for this (class, resolution)."""
        ctx = {"norgate_active": self.norgate_active}
        for rule in self.rules:
            if rule.instrument_class != instrument_class:
                continue
            if rule.resolutions and resolution not in rule.resolutions:
                continue
            for src_rule in rule.priority:
                if _eval_condition(src_rule.condition, ctx):
                    return src_rule.source
        return None

    # ── persistence ──────────────────────────────────────────────────────
    @classmethod
    def default(cls) -> "SourcePriorityConfig":
        """The repo's default priority: Norgate for futures + equities;
        MT5 for CFDs; W/M derived from daily."""
        D = (BarAggregation.DAY,)
        WM = (BarAggregation.WEEK, BarAggregation.MONTH)
        return cls(
            rules=[
                PriorityRule(InstrumentClass.FUTURE, D, (
                    SourceRule("norgate", "norgate_active == True"),
                )),
                PriorityRule(InstrumentClass.FUTURE, WM, (
                    SourceRule("derived_from_daily", "always"),
                )),
                PriorityRule(InstrumentClass.SPOT, D, (
                    SourceRule("norgate", "norgate_active == True"),
                )),
                PriorityRule(InstrumentClass.CFD, (), (
                    SourceRule("mt5", "always"),
                )),
            ],
            norgate_active=True,
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "SourcePriorityConfig":
        """Load from YAML if present, else the built-in default.

        YAML schema (configs/source_priority.yaml):
            norgate_active: true
            source_priority_rules:
              - instrument_class: FUTURE
                resolution: D            # or [W, M]
                priority:
                  - {source: norgate, condition: "norgate_active == True"}
        """
        path = path or _default_config_path()
        if not path.exists():
            return cls.default()
        import yaml  # local import; PyYAML is a light dep

        raw = yaml.safe_load(path.read_text()) or {}
        rules: list[PriorityRule] = []
        for r in raw.get("source_priority_rules", []):
            res = r.get("resolution")
            res_list = res if isinstance(res, list) else ([res] if res else [])
            resolutions = tuple(_RESOLUTION_CODE_TO_AGG[x] for x in res_list)
            priority = tuple(
                SourceRule(p["source"], p.get("condition", "always"))
                for p in r.get("priority", [])
            )
            rules.append(PriorityRule(
                InstrumentClass(r["instrument_class"]), resolutions, priority))
        return cls(rules=rules, norgate_active=bool(raw.get("norgate_active", True)))


def _default_config_path() -> Path:
    here = Path(__file__).resolve()
    root = next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[2],
    )
    return root / "configs" / "source_priority.yaml"
