"""Bar-by-bar futures contract simulation for portfolio research.

Converts ``position_fraction`` signals to integer contract counts using a fixed
account capital budget, then computes discrete-contract PnL and compares it to
the standard fractional-return path to produce tracking-error diagnostics,
tearsheets, and a summary HTML report.

Designed to be called from ``portfolio_research.pipelines.portfolio_test._evaluate_phase``
after ``combined_positions`` and ``daily_test_candles`` are available.

Return convention
-----------------
Both paths (fractional and discrete) use **simple price-change × multiplier**
for contract PnL so the comparison is apples-to-apples at the dollar level.
Daily % returns for tearsheets are derived by dividing each day's USD PnL by
``account_capital``, so the tearsheet y-axis reads in familiar percentage units
with exactly the same format as the standard fractional-return tearsheets.

The fractional path is recomputed here (not taken from the log-return tearsheet
series) to ensure the same instrument-return definition is used for both legs.

Outputs (under ``output_dir / "futures_sim"``)
----------------------------------------------
``{phase}_diagnostics.csv``
    Per-bar per-ticker: ticker, date, position_fraction, price, contracts,
    contract_value, notional, margin_required, margin_available,
    leverage_breach, discrete_pnl, fractional_pnl.

``{phase}_tracking_error.csv``
    Per-date aggregate: discrete_total_pnl, fractional_total_pnl,
    discrete_pct_return, fractional_pct_return, daily_tracking_error_usd,
    daily_tracking_error_pct, cumulative_te_usd, annualised_te_vol_usd,
    any_leverage_breach.

``{phase}_tracking_error_summary.html``
    Self-contained HTML report with a prose summary table, per-ticker
    breakdown, and daily tracking-error chart (matplotlib, base64-embedded).

``{phase}_discrete_tearsheet.html``
    Standard QuantStats tearsheet driven by discrete daily % returns
    (same format as the regular portfolio tearsheets).
"""
from __future__ import annotations

import base64
import io
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from portfolio_research.config import FuturesSimConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_simple_instrument_returns(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(ticker, datetime): close, next_close, simple_return."""
    df = candles_df[["ticker", "datetime", "close"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("s")
    df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    df["next_close"] = df.groupby("ticker")["close"].shift(-1)
    df["simple_return"] = df["next_close"] / df["close"] - 1.0
    return df


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

@dataclass
class SimBar:
    ticker: str
    date: pd.Timestamp
    position_fraction: float
    price: float
    contracts: int
    contract_value: float
    notional: float
    margin_required: float
    margin_available: float
    leverage_breach: bool
    discrete_pnl: float
    fractional_pnl: float


def _simulate_ticker_bars(
    ticker: str,
    ticker_positions: pd.DataFrame,
    instrument_returns: pd.DataFrame,
    multiplier: float,
    margin_long: float,
    margin_short: float,
    account_capital: float,
    finite_leverage: bool,
) -> list[SimBar]:
    """Simulate one ticker's bars; return a list of SimBar records."""
    rets = instrument_returns[instrument_returns["ticker"] == ticker].set_index("datetime")

    positions = ticker_positions.copy()
    positions["datetime"] = pd.to_datetime(positions["datetime"]).dt.floor("s")
    positions = positions.sort_values("datetime")

    bars: list[SimBar] = []
    for row in positions.itertuples(index=False):
        dt = row.datetime
        pos_frac = float(row.position_fraction)

        if dt not in rets.index:
            continue
        ret_row = rets.loc[dt]
        simple_ret = float(ret_row["simple_return"]) if not pd.isna(ret_row["simple_return"]) else float("nan")
        if pd.isna(simple_ret):
            continue

        price = float(ret_row["close"])
        contract_value = price * multiplier

        # sizing
        target_dollars = pos_frac * account_capital
        contracts_raw = target_dollars / contract_value if contract_value > 0 else 0.0
        contracts = int(round(contracts_raw))

        # margin
        notional = abs(contracts) * contract_value
        margin_per_contract = margin_long if contracts >= 0 else margin_short
        margin_required = abs(contracts) * margin_per_contract
        margin_available = account_capital
        leverage_breach = finite_leverage and (margin_required > margin_available)

        # PnL
        price_move = simple_ret * price          # next_close - close
        discrete_pnl = contracts * price_move * multiplier
        fractional_pnl = pos_frac * account_capital * simple_ret

        bars.append(SimBar(
            ticker=ticker,
            date=dt,
            position_fraction=pos_frac,
            price=price,
            contracts=contracts,
            contract_value=contract_value,
            notional=notional,
            margin_required=margin_required,
            margin_available=margin_available,
            leverage_breach=leverage_breach,
            discrete_pnl=discrete_pnl,
            fractional_pnl=fractional_pnl,
        ))

    return bars


# ---------------------------------------------------------------------------
# Tearsheet helpers
# ---------------------------------------------------------------------------

def _usd_pnl_to_pct_returns(daily_pnl: pd.Series, account_capital: float) -> pd.Series:
    """Convert daily USD PnL series to daily % return series suitable for QuantStats."""
    pct = daily_pnl / account_capital
    pct.name = "strategy_return"
    return pct


def _emit_discrete_tearsheet(
    daily_discrete_pnl: pd.Series,
    daily_fractional_pnl: pd.Series,
    account_capital: float,
    phase_name: str,
    sim_dir: Path,
) -> None:
    """Generate a QuantStats tearsheet from discrete daily % returns."""
    try:
        from metrics.plotting.graphing.quantstats_reports import generate_tearsheet
    except ImportError:
        return

    discrete_pct = _usd_pnl_to_pct_returns(daily_discrete_pnl, account_capital)
    fractional_pct = _usd_pnl_to_pct_returns(daily_fractional_pnl, account_capital)

    # Align indices so QuantStats can compare them
    idx = discrete_pct.index.union(fractional_pct.index)
    discrete_pct = discrete_pct.reindex(idx).fillna(0.0)
    fractional_pct = fractional_pct.reindex(idx).fillna(0.0)

    safe_phase = phase_name.replace(" ", "_")
    out_path = sim_dir / f"{safe_phase}_discrete_tearsheet.html"

    generate_tearsheet(
        strategy_returns=discrete_pct,
        baseline_returns=fractional_pct,
        feature_name=f"Discrete Contracts — {phase_name}",
        output_file=str(out_path),
        mode="html",
    )
    print(f"  [futures_sim] discrete tearsheet -> {out_path}")


# ---------------------------------------------------------------------------
# Tracking error summary HTML
# ---------------------------------------------------------------------------

def _build_ticker_summary(diag_df: pd.DataFrame, account_capital: float) -> pd.DataFrame:
    """Per-ticker aggregate statistics for the summary report."""
    rows = []
    for ticker, grp in diag_df.groupby("ticker"):
        n_days = len(grp)
        avg_contracts = float(grp["contracts"].abs().mean())
        max_contracts = int(grp["contracts"].abs().max())
        n_breaches = int(grp["leverage_breach"].sum())
        total_discrete = float(grp["discrete_pnl"].sum())
        total_frac = float(grp["fractional_pnl"].sum())
        te_daily = grp["discrete_pnl"] - grp["fractional_pnl"]
        te_vol = float(te_daily.std()) * np.sqrt(252) if len(te_daily) > 1 else float("nan")
        rows.append({
            "ticker": ticker,
            "trading_days": n_days,
            "avg_abs_contracts": round(avg_contracts, 2),
            "max_abs_contracts": max_contracts,
            "total_discrete_pnl": round(total_discrete, 2),
            "total_fractional_pnl": round(total_frac, 2),
            "cumulative_te_usd": round(total_discrete - total_frac, 2),
            "annualised_te_vol_usd": round(te_vol, 2),
            "margin_breach_days": n_breaches,
        })
    return pd.DataFrame(rows)


def _sparkline_base64(series: pd.Series) -> str:
    """Return a tiny inline PNG of the series as a base64 data URI."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 1.2))
        ax.plot(series.values, linewidth=0.8, color="#1f77b4")
        ax.axhline(0, color="#aaa", linewidth=0.5, linestyle="--")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        fig.tight_layout(pad=0.1)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def _emit_tracking_error_summary(
    phase_name: str,
    daily_te: pd.DataFrame,
    diag_df: pd.DataFrame,
    account_capital: float,
    leverage_mode_label: str,
    instrument_specs_repr: dict[str, str],
    sim_dir: Path,
) -> None:
    """Write a self-contained HTML summary for the tracking-error analysis."""
    safe_phase = phase_name.replace(" ", "_")
    out_path = sim_dir / f"{safe_phase}_tracking_error_summary.html"

    cum_te = float(daily_te["cumulative_te_usd"].iloc[-1])
    mean_daily_te = float(daily_te["daily_tracking_error_usd"].mean())
    te_vol = float(daily_te["annualised_te_vol_usd"].iloc[0]) if "annualised_te_vol_usd" in daily_te.columns else float("nan")
    n_breaches = int(daily_te["any_leverage_breach"].sum())
    n_days = len(daily_te)
    total_discrete = float(daily_te["discrete_total_pnl"].sum())
    total_frac = float(daily_te["fractional_total_pnl"].sum())

    ticker_summary = _build_ticker_summary(diag_df, account_capital)

    # chart
    chart_uri = _sparkline_base64(daily_te.set_index("date")["cumulative_te_usd"])
    chart_tag = f'<img src="{chart_uri}" style="width:400px;height:120px" alt="Cumulative TE">' if chart_uri else ""

    # ticker table rows
    ticker_rows = "".join(
        f"<tr>"
        f"<td>{r.ticker}</td>"
        f"<td>{r.trading_days}</td>"
        f"<td>{r.avg_abs_contracts:.2f}</td>"
        f"<td>{r.max_abs_contracts}</td>"
        f"<td>${r.total_discrete_pnl:,.0f}</td>"
        f"<td>${r.total_fractional_pnl:,.0f}</td>"
        f"<td>${r.cumulative_te_usd:+,.0f}</td>"
        f"<td>${r.annualised_te_vol_usd:,.0f}/yr</td>"
        f"<td>{'⚠ ' + str(r.margin_breach_days) if r.margin_breach_days else '—'}</td>"
        f"</tr>"
        for r in ticker_summary.itertuples(index=False)
    )

    # instrument spec table
    spec_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in instrument_specs_repr.items()
    )

    html = textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <title>Futures Sim Tracking Error — {phase_name}</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              font-size: 13px; color: #222; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
      h1   {{ font-size: 20px; border-bottom: 2px solid #1f77b4; padding-bottom: 6px; }}
      h2   {{ font-size: 15px; color: #333; margin-top: 28px; }}
      table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
      th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
      th   {{ background: #f5f5f5; text-align: center; }}
      td:first-child {{ text-align: left; font-weight: 600; }}
      .ok  {{ color: #2a9d2a; }} .warn {{ color: #c05000; }}
      .note {{ font-size: 11px; color: #666; margin-top: 6px; }}
    </style>
    </head>
    <body>
    <h1>Futures Simulation — Tracking Error Summary<br>
        <small style="font-weight:normal;font-size:14px">Phase: {phase_name} &nbsp;|&nbsp; Leverage: {leverage_mode_label}</small>
    </h1>

    <h2>Portfolio-Level Summary</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Account capital</td><td>${account_capital:,.0f}</td></tr>
      <tr><td>Trading days evaluated</td><td>{n_days}</td></tr>
      <tr><td>Total discrete PnL</td><td>${total_discrete:+,.0f}</td></tr>
      <tr><td>Total fractional PnL</td><td>${total_frac:+,.0f}</td></tr>
      <tr><td>Cumulative tracking error (discrete − fractional)</td>
          <td class="{'ok' if abs(cum_te) < 0.02 * account_capital else 'warn'}">${cum_te:+,.0f}</td></tr>
      <tr><td>Mean daily tracking error</td><td>${mean_daily_te:+,.2f}</td></tr>
      <tr><td>Annualised TE volatility</td>
          <td>{'${:,.0f}/yr'.format(te_vol) if not pd.isna(te_vol) else 'n/a'}</td></tr>
      <tr><td>Days with margin breach {'(FINITE mode)' if leverage_mode_label == 'FINITE' else ''}</td>
          <td class="{'warn' if n_breaches else 'ok'}">{n_breaches}</td></tr>
    </table>

    <h2>Cumulative Tracking Error (USD)</h2>
    {chart_tag if chart_tag else '<p class="note">Install matplotlib to render chart.</p>'}
    <p class="note">Positive = discrete contracts earned more than fractional baseline; negative = rounding cost.</p>

    <h2>Per-Ticker Breakdown</h2>
    <table>
      <tr>
        <th>Ticker</th><th>Days</th><th>Avg |Contracts|</th><th>Max |Contracts|</th>
        <th>Discrete PnL</th><th>Fractional PnL</th>
        <th>Cum TE</th><th>Ann TE Vol</th><th>Margin Breaches</th>
      </tr>
      {ticker_rows}
    </table>

    <h2>Instrument Specs Used</h2>
    <table>
      <tr><th>Ticker</th><th>Spec</th></tr>
      {spec_rows}
    </table>

    <h2>Interpretation Guide</h2>
    <ul>
      <li><b>Tracking error</b> = difference between discrete-contract PnL and the
          equivalent fractional/continuous PnL.  It arises purely from contract rounding
          (e.g. 2.4 → 2 contracts loses 0.4 contracts of exposure each bar).</li>
      <li><b>Annualised TE vol</b> = std(daily TE) × √252.  Compare to total daily PnL
          vol to gauge the rounding penalty as a fraction of strategy risk.</li>
      <li><b>Margin breach</b> = a day where <code>abs(contracts) × margin_per_contract
          &gt; account_capital</code>.  Increase capital or reduce position_fraction
          max to eliminate.</li>
      <li>The <b>discrete tearsheet</b> ({safe_phase}_discrete_tearsheet.html) uses
          daily % returns = daily USD PnL / account_capital, so metrics are directly
          comparable to the standard portfolio tearsheets.</li>
    </ul>
    </body>
    </html>
    """)

    out_path.write_text(html, encoding="utf-8")
    print(f"  [futures_sim] TE summary -> {out_path}")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run_futures_sim(
    phase_name: str,
    combined_positions: pd.DataFrame,
    daily_test_candles: pd.DataFrame,
    sim_config: "FuturesSimConfig",
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Run the futures contract simulation for one portfolio phase.

    Parameters
    ----------
    phase_name:
        Label used in output filenames (e.g. ``"Train"``, ``"Test"``).
    combined_positions:
        DataFrame with columns ``[ticker, datetime, position_fraction]``.
        Same object passed to ``calculate_strategy_returns_from_positions``.
    daily_test_candles:
        Daily candle DataFrame with columns ``[ticker, datetime, close, ...]``.
    sim_config:
        ``FuturesSimConfig`` from ``PortfolioResearchConfig``.
    output_dir:
        Phase output directory (e.g. ``config.output_root / "test"``).

    Returns
    -------
    dict with keys:

    - ``"diagnostics"`` — per-bar per-ticker DataFrame
    - ``"tracking_error"`` — daily aggregate DataFrame
    - ``"discrete_pct_returns"`` — daily % return series (discrete contracts)
    - ``"fractional_pct_returns"`` — daily % return series (fractional baseline)

    All values are empty / empty Series when ``sim_config.enabled`` is False or
    no specs are configured.
    """
    from portfolio_research.config import LeverageMode

    _empty: dict[str, pd.DataFrame | pd.Series] = {
        "diagnostics": pd.DataFrame(),
        "tracking_error": pd.DataFrame(),
        "discrete_pct_returns": pd.Series(dtype=float),
        "fractional_pct_returns": pd.Series(dtype=float),
    }

    if not sim_config.enabled or not sim_config.instrument_specs:
        return _empty  # type: ignore[return-value]

    instrument_returns = _compute_simple_instrument_returns(daily_test_candles)
    finite_leverage = sim_config.leverage_mode == LeverageMode.FINITE

    positions_norm = combined_positions.copy()
    positions_norm["datetime"] = pd.to_datetime(positions_norm["datetime"]).dt.floor("s")
    positions_norm["ticker"] = positions_norm["ticker"].astype(str)

    all_bars: list[SimBar] = []
    for ticker, spec in sim_config.instrument_specs.items():
        ticker_positions = positions_norm[positions_norm["ticker"] == ticker]
        if ticker_positions.empty:
            continue
        bars = _simulate_ticker_bars(
            ticker=ticker,
            ticker_positions=ticker_positions,
            instrument_returns=instrument_returns,
            multiplier=spec.multiplier,
            margin_long=spec.margin_long,
            margin_short=spec.margin_short,
            account_capital=sim_config.account_capital,
            finite_leverage=finite_leverage,
        )
        all_bars.extend(bars)

    if not all_bars:
        return _empty  # type: ignore[return-value]

    diag_df = pd.DataFrame([
        {
            "ticker": b.ticker,
            "date": b.date,
            "position_fraction": b.position_fraction,
            "price": b.price,
            "contracts": b.contracts,
            "contract_value": b.contract_value,
            "notional": b.notional,
            "margin_required": b.margin_required,
            "margin_available": b.margin_available,
            "leverage_breach": b.leverage_breach,
            "discrete_pnl": b.discrete_pnl,
            "fractional_pnl": b.fractional_pnl,
        }
        for b in all_bars
    ])
    diag_df["date"] = pd.to_datetime(diag_df["date"])
    diag_df = diag_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # --- daily aggregates ---
    daily = (
        diag_df.groupby("date")[["discrete_pnl", "fractional_pnl", "leverage_breach"]]
        .agg({"discrete_pnl": "sum", "fractional_pnl": "sum", "leverage_breach": "any"})
        .reset_index()
    )
    daily = daily.rename(columns={
        "discrete_pnl": "discrete_total_pnl",
        "fractional_pnl": "fractional_total_pnl",
        "leverage_breach": "any_leverage_breach",
    })
    capital = sim_config.account_capital
    daily["discrete_pct_return"] = daily["discrete_total_pnl"] / capital
    daily["fractional_pct_return"] = daily["fractional_total_pnl"] / capital
    daily["daily_tracking_error_usd"] = daily["discrete_total_pnl"] - daily["fractional_total_pnl"]
    daily["daily_tracking_error_pct"] = daily["daily_tracking_error_usd"] / capital
    daily["cumulative_te_usd"] = daily["daily_tracking_error_usd"].cumsum()

    n_days = len(daily)
    te_vol = float(daily["daily_tracking_error_usd"].std()) * np.sqrt(252) if n_days > 1 else float("nan")
    daily["annualised_te_vol_usd"] = te_vol

    daily = daily[[
        "date",
        "discrete_total_pnl",
        "fractional_total_pnl",
        "discrete_pct_return",
        "fractional_pct_return",
        "daily_tracking_error_usd",
        "daily_tracking_error_pct",
        "cumulative_te_usd",
        "annualised_te_vol_usd",
        "any_leverage_breach",
    ]]

    # --- output directory ---
    sim_dir = output_dir / "futures_sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    safe_phase = phase_name.replace(" ", "_")

    # --- CSVs ---
    if sim_config.emit_diagnostics_csv:
        diag_path = sim_dir / f"{safe_phase}_diagnostics.csv"
        diag_df.to_csv(diag_path, index=False)
        print(f"  [futures_sim] diagnostics -> {diag_path}")

    if sim_config.emit_tracking_error_csv:
        te_path = sim_dir / f"{safe_phase}_tracking_error.csv"
        daily.to_csv(te_path, index=False)
        n_breaches = int(daily["any_leverage_breach"].sum())
        cum_te = float(daily["cumulative_te_usd"].iloc[-1])
        print(
            f"  [futures_sim] tracking error -> {te_path} "
            f"| cumTE={cum_te:+,.0f} USD | annTE_vol={te_vol:.0f} USD/yr "
            f"| margin_breaches={n_breaches}d"
        )

    # --- tracking error summary HTML ---
    leverage_label = sim_config.leverage_mode.value.upper()
    specs_repr = {
        ticker: (
            f"{spec.product_code} | ×{spec.multiplier} | "
            f"margin L=${spec.margin_long:,.0f} / S=${spec.margin_short:,.0f}"
        )
        for ticker, spec in sim_config.instrument_specs.items()
    }
    _emit_tracking_error_summary(
        phase_name=phase_name,
        daily_te=daily,
        diag_df=diag_df,
        account_capital=capital,
        leverage_mode_label=leverage_label,
        instrument_specs_repr=specs_repr,
        sim_dir=sim_dir,
    )

    # --- discrete tearsheet ---
    daily_discrete_pnl = daily.set_index("date")["discrete_total_pnl"]
    daily_fractional_pnl = daily.set_index("date")["fractional_total_pnl"]
    _emit_discrete_tearsheet(
        daily_discrete_pnl=daily_discrete_pnl,
        daily_fractional_pnl=daily_fractional_pnl,
        account_capital=capital,
        phase_name=phase_name,
        sim_dir=sim_dir,
    )

    # --- return series for callers ---
    discrete_pct = daily.set_index("date")["discrete_pct_return"].rename("strategy_return")
    fractional_pct = daily.set_index("date")["fractional_pct_return"].rename("strategy_return")

    return {
        "diagnostics": diag_df,
        "tracking_error": daily,
        "discrete_pct_returns": discrete_pct,
        "fractional_pct_returns": fractional_pct,
    }
