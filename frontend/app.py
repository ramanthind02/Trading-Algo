from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT_DIR = Path(__file__).resolve().parent.parent
INTRADAY_DIR = ROOT_DIR / "data" / "intraday_adjusted"
DAILY_DIR = ROOT_DIR / "data" / "ohlc_data"
KIBOT_BACKUP_DIR = ROOT_DIR / "data" / "ohlc_data_kibot_backup"

# Tickers not migrated to Norgate — backup is identical to current data
KIBOT_ONLY_TICKERS = {"NG", "TLT"}

INTRADAY_TFS = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10",
    "M15", "M30", "H1", "H2", "H4",
]
DAILY_TFS = ["D", "W", "M"]

app = Flask(__name__, static_folder=".")
CORS(app)


# ── Static file serving ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename: str):
    return send_from_directory(".", filename)


# ── REST endpoints ───────────────────────────────────────────────────

@app.route("/tickers")
def tickers():
    """Return sorted list of tickers that have data in either directory."""
    ticker_set: set[str] = set()
    for d in (INTRADAY_DIR, DAILY_DIR):
        if d.exists():
            ticker_set.update(p.name for p in d.iterdir() if p.is_dir())
    return jsonify(sorted(ticker_set))


@app.route("/timeframes/<ticker>")
def timeframes(ticker: str):
    """Return ordered list of available timeframes for a ticker."""
    available: list[str] = []
    for tf in INTRADAY_TFS:
        path = INTRADAY_DIR / ticker / f"{tf}_{ticker}.parquet"
        if path.exists():
            available.append(tf)
    for tf in DAILY_TFS:
        path = DAILY_DIR / ticker / f"{tf}_{ticker}.parquet"
        if path.exists():
            available.append(tf)
    return jsonify(available)


# ── Candle data endpoint ──────────────────────────────────────────────

MAX_CANDLES = 5000
DEFAULT_CANDLES = 500


def _resolve_parquet_path(ticker: str, tf: str) -> Path | None:
    """Return the parquet file path for a ticker/timeframe, or None."""
    if tf in INTRADAY_TFS:
        p = INTRADAY_DIR / ticker / f"{tf}_{ticker}.parquet"
    elif tf in DAILY_TFS:
        p = DAILY_DIR / ticker / f"{tf}_{ticker}.parquet"
    else:
        return None
    return p if p.exists() else None


@app.route("/candles/<ticker>/<tf>")
def candles(ticker: str, tf: str):
    """Return paginated candle data as JSON.

    Query params
    ------------
    count  : int  – number of candles (default 500, max 5000)
    before : int  – unix timestamp; return candles before this time
    from   : str  – ISO date (YYYY-MM-DD); start of date range
    to     : str  – ISO date (YYYY-MM-DD); end of date range
    """
    path = _resolve_parquet_path(ticker, tf)
    if path is None:
        return jsonify({"error": "not found"}), 404

    df = pd.read_parquet(path)

    date_from = request.args.get("from")
    date_to = request.args.get("to")

    if date_from or date_to:
        # Date-range mode
        if date_from:
            ts_from = int(pd.Timestamp(date_from).timestamp())
            df = df[df["timestamp"] >= ts_from]
        if date_to:
            ts_to = int(pd.Timestamp(date_to + " 23:59:59").timestamp())
            df = df[df["timestamp"] <= ts_to]

        total_in_range = len(df)
        capped = total_in_range > MAX_CANDLES
        if capped:
            df = df.tail(MAX_CANDLES)
    else:
        # Pagination mode (most-recent-N or before-cursor)
        count = min(int(request.args.get("count", DEFAULT_CANDLES)), MAX_CANDLES)
        before = request.args.get("before")
        total_in_range = len(df)

        if before:
            df = df[df["timestamp"] < int(before)]

        capped = False
        df = df.tail(count)

    records = [
        {
            "time": int(row["timestamp"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for row in df.to_dict("records")
    ]

    earliest_ts = records[0]["time"] if records else None
    has_more = earliest_ts is not None and earliest_ts > _first_timestamp(path)

    resp: dict = {
        "candles": records,
        "total_available": total_in_range,
        "has_more": has_more,
        "earliest_timestamp": earliest_ts,
    }
    if date_from or date_to:
        resp["capped"] = capped
        if capped:
            resp["cap"] = MAX_CANDLES

    return jsonify(resp)


@app.route("/candles/<ticker>/<tf>/kibot")
def candles_kibot(ticker: str, tf: str):
    """Return Kibot backup candle data for comparison overlay."""
    if tf not in DAILY_TFS:
        return jsonify({"error": "Kibot backup only available for D/W/M"}), 400

    if ticker in KIBOT_ONLY_TICKERS:
        return jsonify({"error": "Ticker was not migrated to Norgate"}), 404

    path = KIBOT_BACKUP_DIR / ticker / f"{tf}_{ticker}.parquet"
    if not path.exists():
        return jsonify({"error": "No Kibot backup for this ticker/tf"}), 404

    df = pd.read_parquet(path)

    # Apply same date filtering as main candles endpoint
    date_from = request.args.get("from")
    date_to = request.args.get("to")
    if date_from:
        ts_from = int(pd.Timestamp(date_from).timestamp())
        df = df[df["timestamp"] >= ts_from]
    if date_to:
        ts_to = int(pd.Timestamp(date_to + " 23:59:59").timestamp())
        df = df[df["timestamp"] <= ts_to]

    count = min(int(request.args.get("count", MAX_CANDLES)), MAX_CANDLES)
    before = request.args.get("before")
    if before:
        df = df[df["timestamp"] < int(before)]
    df = df.tail(count)

    records = [
        {"time": int(row["timestamp"]), "value": float(row["close"])}
        for row in df.to_dict("records")
    ]
    return jsonify({"candles": records})


def _first_timestamp(path: Path) -> int:
    """Return the first timestamp in a parquet file (cheap read)."""
    df_head = pd.read_parquet(path, columns=["timestamp"]).head(1)
    return int(df_head.iloc[0]["timestamp"])


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5001)
