"""Extract intraday parquet files from kibot_data.zip.

Extracts all minute (M1-M30) and hour (H1-H4) timeframes.
Excludes seconds (S1/S5/S15/S30) and daily+ (D/W/M) data.
D files are already tracked in git with longer history and volume.

Output structure:
    data/intraday_original/{TICKER}/{TF}_{TICKER}.parquet

Usage:
    python scripts/extract_kibot_data.py
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

# Timeframe prefixes to extract (minutes + hours, no seconds or daily+)
_INTRADAY_PATTERN = re.compile(r"^(M\d+|H\d+)$")
_SKIP = {"D", "W", "M", "S1", "S5", "S15", "S30"}


def main() -> None:
    zip_path = Path("data/kibot_data.zip")
    output_dir = Path("data/intraday_original")

    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Place the Kibot zip in data/ first."
        )

    extracted = 0
    with zipfile.ZipFile(zip_path) as z:
        for entry in z.infolist():
            if not entry.filename.endswith(".parquet"):
                continue

            # Parse: ohlc_data/{TICKER}/{TF}_{TICKER}.parquet
            parts = entry.filename.split("/")
            if len(parts) < 3:
                continue

            ticker = parts[1]
            filename = parts[2]
            tf = filename.replace(f"_{ticker}.parquet", "")

            if tf in _SKIP or not _INTRADAY_PATTERN.match(tf):
                continue

            ticker_dir = output_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            dest = ticker_dir / filename

            with z.open(entry) as src:
                dest.write_bytes(src.read())

            extracted += 1
            print(f"  {ticker}/{filename} ({entry.file_size / 1e6:.1f} MB)")

    print(f"\nExtracted {extracted} files to {output_dir}/")


if __name__ == "__main__":
    main()
