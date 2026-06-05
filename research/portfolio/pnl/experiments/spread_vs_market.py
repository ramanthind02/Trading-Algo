"""WP-3 realism experiment — spreads vs market orders on the NDX 2026 lane.

ADDITIVE / opt-in. Compares entry :class:`ExecutionPolicy` variants on the
NDX 2026 MT5 intraday reference data using the Nautilus realism lane
(:class:`research.portfolio.pnl.nautilus_engine.NautilusPnLEngine`):

* ``MARKET_ON_OPEN``   — cross the spread at the open (taker).
* ``LIMIT_AT_TOUCH``   — rest at the near touch (maker; capture spread if filled).
* ``LIMIT_IMPROVE(n)`` — rest *inside* the spread by ``n`` ticks.

All limit policies use a ``CROSS_AFTER`` fallback (convert to market if unfilled
by a session-fraction cutoff) so the daily target is always reached.

The study isolates *execution* from signal noise by driving a **constant daily
``position_fraction`` target** (default ``+1`` long). Because the raw NDX 2026
tick history is ~54M rows, the harness slices a small **session window** and
ingests bars + bid/ask quotes into a throwaway catalog before running each
policy. The reported quantities per policy:

* **fill rate** — entry fills as a fraction of session targets.
* **avg signed spread vs mid** — positive = spread *captured* (maker at touch),
  negative = spread *paid* (taker cross). Price units and basis points of mid.
* **MAKER / TAKER split** — from Nautilus ``OrderFilled.liquidity_side``.
* **total-return impact** — sum of the lane's per-bar log returns.

Run::

    .\\.venv\\Scripts\\python.exe -m research.portfolio.pnl.experiments.spread_vs_market

Writes a markdown summary and a CSV to ``--out-dir`` (default this package's
``results/``).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_platform.nautilus.ingest import (
    _build_bars,
    _build_quotes,
    _resolve_instrument,
    mt5_data_root,
)
from data_platform.nautilus.instruments import to_nautilus_instrument
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId as NTInstrumentId
from research.portfolio.pnl.nautilus_engine import (
    CrossAfterPolicy,
    ExecutionPolicy,
    ExecutionWindowPolicy,
    FillDiagnostic,
    LaneResult,
    NautilusPnLEngine,
)
from data_platform.nautilus.catalog import get_catalog

_SYMBOL = "NDX"


# ---------------------------------------------------------------------------
# Sliced-catalog construction (keep the run tractable vs the 54M-tick history)
# ---------------------------------------------------------------------------


def _read_partition_window(
    base: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    parts = sorted(base.glob("year=*/part.parquet"))
    frames = [pd.read_parquet(p) for p in parts]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(df["time"], utc=True)
    return df[(t >= start) & (t < end)].reset_index(drop=True)


def build_sliced_catalog(
    symbol: str, start: pd.Timestamp, end: pd.Timestamp, catalog_path: str
):
    """Ingest bars + quotes for *symbol* over [start, end) into a fresh catalog."""
    inst = _resolve_instrument(symbol)
    nt_inst = to_nautilus_instrument(inst)
    instrument_id = NTInstrumentId.from_str(str(inst.id))
    pp = inst.price_precision
    bar_type = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL")

    sym_root = mt5_data_root() / symbol
    bars_df = _read_partition_window(sym_root / "bars_M1", start, end)
    ticks_df = _read_partition_window(sym_root / "ticks", start, end)
    if bars_df.empty or ticks_df.empty:
        raise FileNotFoundError(
            f"No NDX bars/ticks in window {start}..{end} "
            f"(bars={len(bars_df)}, ticks={len(ticks_df)})."
        )

    bars = _build_bars(bars_df, bar_type, pp)
    quotes = _build_quotes(ticks_df, instrument_id, pp)

    catalog = get_catalog(catalog_path)
    catalog.write_data(bars)
    catalog.write_data(quotes)
    return bars_df, ticks_df


def constant_long_targets(bars_df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """One ``position_fraction`` target per session date in the bar window."""
    dates = pd.to_datetime(bars_df["time"], utc=True).dt.normalize().unique()
    return pd.DataFrame(
        {
            "ticker": _SYMBOL,
            "datetime": dates,
            "forecast_score": fraction,
            "position_fraction": fraction,
        }
    )


# ---------------------------------------------------------------------------
# Per-policy measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyRow:
    policy: str
    n_sessions: int
    n_entry_fills: int
    n_entry_rejects: int
    fill_rate: float
    maker_fills: int
    taker_fills: int
    # Liquidity-signed half-spread (robust): + captured (MAKER) / - paid (TAKER).
    avg_signed_spread_px: float
    avg_signed_spread_bps: float
    total_log_return: float


def _entry_diags(diags: list[FillDiagnostic]) -> list[FillDiagnostic]:
    return [d for d in diags if d.is_entry]


def summarize(policy_label: str, n_sessions: int, result: LaneResult) -> PolicyRow:
    entries = _entry_diags(result.fill_diagnostics)
    n_fills = len(entries)
    n_rejects = result.entry_rejects
    maker = sum(1 for d in entries if d.liquidity_side == "MAKER")
    taker = sum(1 for d in entries if d.liquidity_side == "TAKER")
    if entries:
        # Liquidity-signed half-spread: robust to bar-vs-quote price skew (keyed
        # on MAKER/TAKER), unlike the raw fill-price-vs-mid number.
        avg_px = sum(d.liquidity_signed_spread for d in entries) / n_fills
        avg_bps = (
            sum(
                d.liquidity_signed_spread / d.mid_at_fill * 1e4
                for d in entries
                if d.mid_at_fill
            )
            / n_fills
        )
    else:
        avg_px = avg_bps = 0.0
    # Fill rate counts sessions where the entry actually filled (rejects/unfilled
    # passive limits are fill risk).
    return PolicyRow(
        policy=policy_label,
        n_sessions=n_sessions,
        n_entry_fills=n_fills,
        n_entry_rejects=n_rejects,
        fill_rate=n_fills / n_sessions if n_sessions else 0.0,
        maker_fills=maker,
        taker_fills=taker,
        avg_signed_spread_px=avg_px,
        avg_signed_spread_bps=avg_bps,
        total_log_return=float(result.returns.sum()) if len(result.returns) else 0.0,
    )


@dataclass(frozen=True)
class PolicySpec:
    label: str
    policy: ExecutionPolicy
    improve_ticks: int = 1
    # Pure-passive measurement of fill economics: no CROSS_AFTER so unfilled
    # limits are *not* converted to taker market orders (isolates maker capture
    # vs fill risk). Set session_fraction < 1.0 to enable the fallback.
    cross_after: CrossAfterPolicy = CrossAfterPolicy(session_fraction=1.0)


def run_policy(
    spec: PolicySpec,
    targets: pd.DataFrame,
    catalog_path: str,
    n_sessions: int,
) -> PolicyRow:
    engine = NautilusPnLEngine(
        window_policy=ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE,
        execution_policy=spec.policy,
        improve_ticks=spec.improve_ticks,
        cross_after=spec.cross_after,
        # Always subscribe to quotes so the MARKET taker's paid half-spread is
        # measured against the mid (limit policies subscribe regardless).
        measure_spread=True,
        catalog_path=catalog_path,
        max_ticks=None,  # quotes already in the sliced catalog
    )
    result = engine.run_with_diagnostics(targets, pd.DataFrame())
    return summarize(spec.label, n_sessions, result)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def default_policies() -> list[PolicySpec]:
    return [
        PolicySpec("MARKET_ON_OPEN", ExecutionPolicy.MARKET_ON_OPEN),
        PolicySpec("LIMIT_AT_TOUCH", ExecutionPolicy.LIMIT_AT_TOUCH),
        PolicySpec("LIMIT_IMPROVE(1)", ExecutionPolicy.LIMIT_IMPROVE, improve_ticks=1),
        PolicySpec("LIMIT_IMPROVE(2)", ExecutionPolicy.LIMIT_IMPROVE, improve_ticks=2),
    ]


def run_experiment(
    start: pd.Timestamp,
    end: pd.Timestamp,
    fraction: float,
    catalog_root: Path,
) -> pd.DataFrame:
    rows: list[PolicyRow] = []
    for spec in default_policies():
        # One catalog per policy (Nautilus is a global singleton per process and
        # the catalog must hold the quote stream for each isolated run).
        cat_path = str(catalog_root / spec.label.replace("(", "_").replace(")", ""))
        bars_df, _ = build_sliced_catalog(_SYMBOL, start, end, cat_path)
        targets = constant_long_targets(bars_df, fraction)
        n_sessions = len(targets)
        rows.append(run_policy(spec, targets, cat_path, n_sessions))
    return pd.DataFrame([r.__dict__ for r in rows])


def _to_markdown(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    market_row = df[df["policy"] == "MARKET_ON_OPEN"]
    market_ret = float(market_row["total_log_return"].iloc[0]) if len(market_row) else 0.0
    lines = [
        "# Spread vs Market — NDX 2026 realism lane",
        "",
        f"- Window: `{start.isoformat()}` .. `{end.isoformat()}` (UTC)",
        "- Lane: `NautilusPnLEngine`, `INTRADAY_OPEN_TO_CLOSE`, constant +1 long target",
        "- Limit policies measured **pure-passive** (no CROSS_AFTER) to isolate",
        "  maker spread capture vs fill risk.",
        "- Spread = **liquidity-signed half-spread** (keyed on Nautilus",
        "  `OrderFilled.liquidity_side`): **+ = captured** (MAKER), **- = paid**",
        "  (TAKER). Robust to the MT5 bar-vs-quote price skew.",
        "",
        "| Policy | Sessions | Entry fills | Rejects | Fill rate | MAKER | TAKER | "
        "Avg spread (px) | Avg spread (bps) | Total log-ret | Δ vs MARKET |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        delta = float(r["total_log_return"]) - market_ret
        lines.append(
            f"| {r['policy']} | {int(r['n_sessions'])} | {int(r['n_entry_fills'])} | "
            f"{int(r['n_entry_rejects'])} | "
            f"{r['fill_rate']:.2%} | {int(r['maker_fills'])} | {int(r['taker_fills'])} | "
            f"{r['avg_signed_spread_px']:+.4f} | {r['avg_signed_spread_bps']:+.3f} | "
            f"{r['total_log_return']:+.6f} | {delta:+.6f} |"
        )
    lines += [
        "",
        "## Reading the table",
        "",
        "- **MARKET_ON_OPEN** crosses the spread: every entry is a TAKER fill with a",
        "  *negative* signed spread (pays ~half-spread vs mid). Fill rate is 100%.",
        "- **LIMIT_AT_TOUCH** rests at the near touch: fills are MAKER with a",
        "  *non-negative* signed spread (captures ~half-spread) but only when the",
        "  market trades to the touch — so fill rate < 100% (the rest is fill risk).",
        "- **LIMIT_IMPROVE(n)** rests inside the spread: higher fill rate than",
        "  at-touch, less spread captured per fill.",
        "- The economics are directionally sane iff LIMIT_AT_TOUCH's avg signed",
        "  spread `>= 0 >=` MARKET_ON_OPEN's (maker captures, taker pays).",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-01-05", help="UTC window start (date)")
    ap.add_argument("--end", default="2026-01-09", help="UTC window end (exclusive)")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "results"),
    )
    ap.add_argument("--catalog-root", default=None)
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_root = (
        Path(args.catalog_root)
        if args.catalog_root
        else out_dir / "_catalogs"
    )

    df = run_experiment(start, end, args.fraction, catalog_root)
    csv_path = out_dir / "spread_vs_market_ndx_2026.csv"
    md_path = out_dir / "spread_vs_market_ndx_2026.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown(df, start, end), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nWrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
