from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT_DIR = Path(__file__).resolve().parent.parent
INTRADAY_DIR = ROOT_DIR / "data" / "intraday_adjusted"
DAILY_DIR = ROOT_DIR / "data" / "ohlc_data"
INTRADAY_RAW_DIR = ROOT_DIR / "data" / "intraday_original"

INTRADAY_TFS = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10",
    "M15", "M30", "H1", "H2", "H4",
]
DAILY_TFS = ["D", "W", "M"]

app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename: str):
    return send_from_directory(".", filename)


@app.route("/tickers")
def tickers():
    ticker_set: set[str] = set()
    for directory in (INTRADAY_DIR, DAILY_DIR):
        if directory.exists():
            ticker_set.update(path.name for path in directory.iterdir() if path.is_dir())
    return jsonify(sorted(ticker_set))


@app.route("/timeframes/<ticker>")
def timeframes(ticker: str):
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


MAX_CANDLES = 5000
DEFAULT_CANDLES = 500


def _resolve_parquet_path(ticker: str, tf: str) -> Path | None:
    if tf in INTRADAY_TFS:
        path = INTRADAY_DIR / ticker / f"{tf}_{ticker}.parquet"
    elif tf in DAILY_TFS:
        path = DAILY_DIR / ticker / f"{tf}_{ticker}.parquet"
    else:
        return None
    return path if path.exists() else None


@app.route("/candles/<ticker>/<tf>")
def candles(ticker: str, tf: str):
    path = _resolve_parquet_path(ticker, tf)
    if path is None:
        return jsonify({"error": "not found"}), 404

    df = pd.read_parquet(path)

    date_from = request.args.get("from")
    date_to = request.args.get("to")

    if date_from or date_to:
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

    response: dict[str, object] = {
        "candles": records,
        "total_available": total_in_range,
        "has_more": has_more,
        "earliest_timestamp": earliest_ts,
    }
    if date_from or date_to:
        response["capped"] = capped
        if capped:
            response["cap"] = MAX_CANDLES

    return jsonify(response)


@app.route("/candles/<ticker>/<tf>/raw_overlay")
def candles_raw_overlay(ticker: str, tf: str):
    """Return optional intraday raw overlay for visual QA."""
    if tf not in INTRADAY_TFS:
        return jsonify({"error": "Raw overlay is intraday-only"}), 400

    path = INTRADAY_RAW_DIR / ticker / f"{tf}_{ticker}.parquet"
    if not path.exists():
        return jsonify({"error": "No raw overlay data for this ticker/timeframe"}), 404

    df = pd.read_parquet(path)

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
    df_head = pd.read_parquet(path, columns=["timestamp"]).head(1)
    return int(df_head.iloc[0]["timestamp"])


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5001)
