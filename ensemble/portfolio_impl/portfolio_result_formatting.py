from __future__ import annotations

from typing import Any, Dict, Union

import pandas as pd


def deep_copy_result(obj: Any) -> Any:
    """Recursively copy DataFrames nested inside dict/list result payloads."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        return {key: deep_copy_result(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [deep_copy_result(item) for item in obj]
    return obj


def format_portfolio_result(
    cached_result: Dict[str, Any],
    *,
    return_ensemble_predictions: bool,
    return_base_model_predictions: bool,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Format cached portfolio payloads according to requested detail level."""
    if return_ensemble_predictions or return_base_model_predictions:
        result: Dict[str, Any] = {"portfolio": deep_copy_result(cached_result["portfolio"])}
        if return_ensemble_predictions:
            result["ensembles"] = deep_copy_result(cached_result.get("ensembles", {}))
        if return_base_model_predictions:
            result["base_models"] = deep_copy_result(cached_result.get("base_models", {}))
        return result
    return deep_copy_result(cached_result["portfolio"])
