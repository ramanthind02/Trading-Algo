"""Normalize ticker identifiers for joins and map keys."""

from __future__ import annotations


def normalize_ticker_key(ticker_val: object) -> str:
    """Normalize ticker identifiers (enum, string, or object with name/value) to a string key."""
    if hasattr(ticker_val, "name"):
        return str(getattr(ticker_val, "name"))
    if hasattr(ticker_val, "value"):
        return str(getattr(ticker_val, "value"))
    if isinstance(ticker_val, str):
        return ticker_val.replace("Ticker.", "")
    return str(ticker_val)
