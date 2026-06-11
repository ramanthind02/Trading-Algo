"""Browse vault features and turn one back into a runnable spec for re-research.

The vault stores each promoted feature as ``vault[_personal]/<TF>/<sleeve>/<ensemble>/`` with an
``ensemble_config.json`` (timeframe, direction, tickers, timestamps) and ``features/<col>.json``
(the model file, which carries the ``bias_node_spec`` = module + params + timeframe).

Past research artifacts were not archived alongside the vault, so "examine past results" is
offered as **re-research**: reconstruct a :class:`StrategySpec` from the stored config and run it
through the normal workbench (exploration/results), which is the faithful way to regenerate them.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from frontend.api.paths import REPO_ROOT, repo_relative

_VAULT_ROOTS: dict[str, Path] = {
    "prop": REPO_ROOT / "vault",
    "personal": REPO_ROOT / "vault_personal",
}
_DAILY_TF = {"D", "W", "M"}


def _coerce_tickers(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and "name" in item:
            out.append(str(item["name"]))
    return out


def _ensemble_summary(profile: str, root: Path, cfg_path: Path) -> dict[str, Any]:
    data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    ens_dir = cfg_path.parent
    parts = ens_dir.relative_to(root).parts  # (timeframe, sleeve, ensemble)
    feature_files = sorted(p.name for p in (ens_dir / "features").glob("*.json")) if (ens_dir / "features").is_dir() else []
    return {
        "profile": profile,
        "ensemble_name": data.get("ensemble_name", ens_dir.name),
        "timeframe": data.get("timeframe") or (parts[0] if parts else None),
        "sleeve": parts[1] if len(parts) >= 2 else None,
        "direction": data.get("direction"),
        "tickers": _coerce_tickers(data.get("tickers")),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "path": repo_relative(ens_dir),
        "feature_count": len(feature_files),
    }


def list_vault(profile: str = "prop") -> dict[str, Any]:
    """Catalog every ensemble in the chosen vault profile."""

    root = _VAULT_ROOTS.get(profile)
    if root is None or not root.exists():
        return {"profile": profile, "features": []}
    features = [
        _ensemble_summary(profile, root, cfg)
        for cfg in root.rglob("ensemble_config.json")
    ]
    features.sort(key=lambda f: (f["timeframe"] or "", f["sleeve"] or "", f["ensemble_name"]))
    return {"profile": profile, "features": features}


def _resolve_ensemble_dir(profile: str, rel_path: str) -> tuple[Path, Path]:
    root = _VAULT_ROOTS.get(profile)
    if root is None:
        raise ValueError(f"Unknown vault profile: {profile}")
    candidate = Path(rel_path)
    ens_dir = (candidate if candidate.is_absolute() else REPO_ROOT / rel_path).resolve()
    if root.resolve() not in ens_dir.parents and ens_dir != root.resolve():
        raise ValueError("Path is outside the vault profile root.")
    if not (ens_dir / "ensemble_config.json").is_file():
        raise FileNotFoundError(f"No ensemble_config.json under {rel_path}")
    return root, ens_dir


def _first_feature(ens_dir: Path) -> dict[str, Any]:
    feature_files = sorted((ens_dir / "features").glob("*.json")) if (ens_dir / "features").is_dir() else []
    if not feature_files:
        raise FileNotFoundError("No feature files under this ensemble.")
    return json.loads(feature_files[0].read_text(encoding="utf-8-sig"))


def feature_detail(profile: str, rel_path: str) -> dict[str, Any]:
    """Ensemble config + the bias_node_spec(s) of its feature files."""

    _root, ens_dir = _resolve_ensemble_dir(profile, rel_path)
    cfg = json.loads((ens_dir / "ensemble_config.json").read_text(encoding="utf-8-sig"))
    features: list[dict[str, Any]] = []
    if (ens_dir / "features").is_dir():
        for fpath in sorted((ens_dir / "features").glob("*.json")):
            data = json.loads(fpath.read_text(encoding="utf-8-sig"))
            features.append(
                {
                    "feature_name": data.get("feature_name", fpath.stem),
                    "bias_node_spec": data.get("bias_node_spec"),
                }
            )
    return {"profile": profile, "path": rel_path, "config": cfg, "features": features}


def spec_from_feature(profile: str, rel_path: str) -> dict[str, Any]:
    """Reconstruct a StrategySpec dict from a vault feature (single-value param grid)."""

    root, ens_dir = _resolve_ensemble_dir(profile, rel_path)
    cfg = json.loads((ens_dir / "ensemble_config.json").read_text(encoding="utf-8-sig"))
    feature = _first_feature(ens_dir)
    bns = feature.get("bias_node_spec") or {}
    timeframes = bns.get("timeframes") or [cfg.get("timeframe", "D")]
    timeframe = timeframes[0] if timeframes else "D"
    sleeve_parts = ens_dir.relative_to(root).parts
    sleeve = sleeve_parts[1] if len(sleeve_parts) >= 2 else "mean_reversion_indices"
    params = bns.get("params", {})
    param_grid = {key: [value] for key, value in params.items()} or {"_": [0]}
    ensemble_name = cfg.get("ensemble_name", ens_dir.name)

    return {
        "spec_version": "1.0",
        "name": f"{ensemble_name}_rerun",
        "hypothesis": f"Re-research of vault feature '{ensemble_name}' ({profile} vault).",
        "author": "vault",
        "created": datetime.now().isoformat(),
        "tickers": _coerce_tickers(cfg.get("tickers")),
        "data_feed": "darwinex_cfd",
        "mode": "daily" if timeframe in _DAILY_TF else "intraday",
        "timeframe": timeframe,
        "windows": None,
        "signal": {"module_name": bns.get("module_name", ""), "param_grid": param_grid},
        "direction": cfg.get("direction", "long_short"),
        "vol_scaling": "blended",
        "vol_scaling_model": "inherit_daily",
        "risk": {"target_vol": 0.15, "forecast_cap": 2.0, "max_position_pct": 3.5, "buffer_fraction": 0.0},
        "account": {"capital": 50000.0, "prop_constraints": None},
        "execution": {
            "entry_policy": "market_on_open",
            "exit_policy": "market_on_open",
            "unfilled_limit": "cross_after",
            "holding": "overnight",
            "fill_feed": "derived",
        },
        "vault": {"weight_hierarchy_group": sleeve, "ensemble_name": ensemble_name},
    }
