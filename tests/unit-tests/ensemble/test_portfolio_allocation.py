from __future__ import annotations

import json

import pytest

from ensemble.portfolio_impl.portfolio_allocation import (
    effective_instrument_weights,
    load_sector_allocation_config,
    resolve_sector_allocation,
)


def test_load_sector_allocation_config_and_resolve_weights(tmp_path) -> None:
    config_path = tmp_path / "sector.json"
    config_path.write_text(
        json.dumps(
            {
                "weight": 1.0,
                "children": [
                    {
                        "weight": 2.0,
                        "tickers": ["ES", "NQ"],
                    },
                    {
                        "weight": 1.0,
                        "tickers": ["GC"],
                        "ticker_weights": {"GC": 1.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_sector_allocation_config(str(config_path))
    resolved = resolve_sector_allocation(config)

    assert resolved == {"ES": 1 / 3, "NQ": 1 / 3, "GC": 1 / 3}


def test_load_sector_allocation_config_rejects_duplicate_tickers(tmp_path) -> None:
    config_path = tmp_path / "sector.json"
    config_path.write_text(
        json.dumps(
            {
                "weight": 1.0,
                "children": [
                    {"weight": 1.0, "tickers": ["ES"]},
                    {"weight": 1.0, "tickers": ["ES"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate ticker in sector allocation config"):
        load_sector_allocation_config(str(config_path))


def test_effective_instrument_weights_uses_residual_fallback() -> None:
    weights = effective_instrument_weights(
        ["ES", "NQ", "GC"],
        {"ES": 0.5, "NQ": 0.3},
    )

    assert weights["ES"] == pytest.approx(0.5)
    assert weights["NQ"] == pytest.approx(0.3)
    assert weights["GC"] == pytest.approx(0.2)
