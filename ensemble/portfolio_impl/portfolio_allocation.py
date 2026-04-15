from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set


def load_sector_allocation_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a sector allocation configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"Sector allocation configuration file not found: {config_path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in sector allocation configuration file: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Sector allocation root must be a JSON object")

    validate_sector_allocation_node(config, seen_tickers=set(), node_path="root")
    return config


def validate_sector_allocation_node(
    node: Dict[str, Any],
    *,
    seen_tickers: Set[str],
    node_path: str,
) -> None:
    """Validate node schema recursively before resolution."""
    weight = node.get("weight")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
        raise ValueError(f"Node '{node_path}' weight must be > 0")

    has_children = "children" in node
    has_tickers = "tickers" in node
    if has_children == has_tickers:
        raise ValueError(
            f"Node '{node_path}' must define exactly one of 'children' or 'tickers'"
        )

    if has_children:
        children = node["children"]
        if not isinstance(children, list) or len(children) == 0:
            raise ValueError(f"Node '{node_path}' children must be a non-empty list")
        for idx, child in enumerate(children):
            if not isinstance(child, dict):
                raise ValueError(f"Node '{node_path}.children[{idx}]' must be an object")
            validate_sector_allocation_node(
                child,
                seen_tickers=seen_tickers,
                node_path=f"{node_path}.children[{idx}]",
            )
        return

    tickers = node["tickers"]
    if not isinstance(tickers, list) or len(tickers) == 0:
        raise ValueError(f"Node '{node_path}' must define at least one ticker")
    if not all(isinstance(ticker, str) and ticker for ticker in tickers):
        raise ValueError(f"Node '{node_path}' tickers must be non-empty strings")
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"Node '{node_path}' contains duplicate tickers within a leaf")

    duplicates = [ticker for ticker in tickers if ticker in seen_tickers]
    if duplicates:
        raise ValueError(f"Duplicate ticker in sector allocation config: {duplicates[0]}")
    seen_tickers.update(tickers)

    ticker_weights = node.get("ticker_weights")
    if ticker_weights is None:
        return

    if not isinstance(ticker_weights, dict):
        raise ValueError(f"Node '{node_path}' ticker_weights must be an object")
    if set(ticker_weights.keys()) != set(tickers):
        raise ValueError(
            f"Node '{node_path}' ticker_weights keys must match tickers exactly"
        )
    for ticker, ticker_weight in ticker_weights.items():
        if (
            not isinstance(ticker_weight, (int, float))
            or isinstance(ticker_weight, bool)
            or ticker_weight <= 0
        ):
            raise ValueError(
                f"Node '{node_path}' ticker_weights values must be > 0 (ticker={ticker})"
            )


def _resolve_sector_allocation_node(
    node: Dict[str, Any],
    *,
    parent_contribution: float,
    resolved: Dict[str, float],
) -> None:
    """Recursively accumulate ticker contributions from a validated node tree."""
    if "children" in node:
        children = node["children"]
        total_weight = sum(child["weight"] for child in children)
        for child in children:
            contribution = parent_contribution * (child["weight"] / total_weight)
            _resolve_sector_allocation_node(
                child,
                parent_contribution=contribution,
                resolved=resolved,
            )
        return

    tickers = node["tickers"]
    ticker_weights = node.get("ticker_weights")
    if ticker_weights is None:
        equal_share = parent_contribution / len(tickers)
        for ticker in tickers:
            resolved[ticker] = resolved.get(ticker, 0.0) + equal_share
        return

    total_ticker_weight = sum(ticker_weights[ticker] for ticker in tickers)
    for ticker in tickers:
        ticker_share = parent_contribution * (ticker_weights[ticker] / total_ticker_weight)
        resolved[ticker] = resolved.get(ticker, 0.0) + ticker_share


def resolve_sector_allocation(config: Dict[str, Any]) -> Dict[str, float]:
    """Resolve sector tree into normalized ticker->weight mapping."""
    resolved: Dict[str, float] = {}

    if "children" in config:
        children = config["children"]
        total_weight = sum(child["weight"] for child in children)
        for child in children:
            contribution = child["weight"] / total_weight
            _resolve_sector_allocation_node(
                child,
                parent_contribution=contribution,
                resolved=resolved,
            )
    elif "tickers" in config:
        _resolve_sector_allocation_node(config, parent_contribution=1.0, resolved=resolved)
    else:
        raise ValueError("Sector allocation root must define exactly one of 'children' or 'tickers'")

    total_resolved = sum(resolved.values())
    if total_resolved <= 0:
        raise ValueError("Resolved sector allocation produced zero total weight")
    return {
        ticker: weight / total_resolved
        for ticker, weight in resolved.items()
    }


def effective_instrument_weights(
    tickers: List[str],
    configured_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Return effective ticker weights with equal-weight or residual fallback."""
    unique_tickers = list(dict.fromkeys(tickers))
    if not unique_tickers:
        return {}

    if configured_weights is None:
        equal_weight = 1.0 / len(unique_tickers)
        return {ticker: equal_weight for ticker in unique_tickers}

    missing_tickers = [ticker for ticker in unique_tickers if ticker not in configured_weights]
    used_weight = sum(
        configured_weights[ticker] for ticker in unique_tickers if ticker in configured_weights
    )
    remaining_weight = max(1.0 - used_weight, 0.0)
    fallback_weight = remaining_weight / len(missing_tickers) if missing_tickers else 0.0
    return {
        ticker: configured_weights.get(ticker, fallback_weight)
        for ticker in unique_tickers
    }
