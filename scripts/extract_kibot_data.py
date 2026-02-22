"""Extract M1 intraday parquet files from kibot_data.zip.

Does NOT touch D_{TICKER}.parquet files — those are tracked in git
with longer history and volume data.

Usage:
    python scripts/extract_kibot_data.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path


def main() -> None:
    zip_path = Path("data/kibot_data.zip")
    output_dir = Path("data/intraday_1min_original")

    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Place the Kibot zip in data/ first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        for entry in z.infolist():
            name = entry.filename
            if "/M1_" not in name or not name.endswith(".parquet"):
                continue

            ticker = name.split("/")[-1].replace("M1_", "").replace(".parquet", "")
            dest = output_dir / f"{ticker}.parquet"

            with z.open(name) as src:
                dest.write_bytes(src.read())

            print(f"  {ticker} -> {dest} ({entry.file_size / 1e6:.1f} MB)")

    print(f"\nDone. Files in {output_dir}/")


if __name__ == "__main__":
    main()
