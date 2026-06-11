# research/specs — strategy specs (round-trippable JSON)

Each `*.json` file here is one **`StrategySpec`** (see `research/spec/strategy_spec.py`) — the flat,
self-validating description of a strategy to research.

This directory is the **shared contract** between the two ways a spec gets written:

- an **agent** (in a Claude Code chat) helps you design a strategy and writes the JSON here, and
- the **frontend Spec Builder** loads the JSON into a form, lets you edit it, and saves it back.

Both paths go through `research/spec/serialization.py`, so a spec written by either is validated
identically on load (grid ≤ 300 combos, window ordering, sleeve membership, fill-feed consistency,
mode↔timeframe).

## Use from Python

```python
from research.spec import load_spec, save_spec, to_feature_config

spec = load_spec("research/specs/es_double7s_mr.json")   # validates on load
cfg  = to_feature_config(spec)                            # adapter → existing pipeline config
```

## Use from the frontend

The frontend API reads/writes these files (`GET/PUT /api/specs/...`) and runs them through the
adapter. Nothing here is written to the vault automatically — promotion is a separate, human-gated
step in the UI.

`es_double7s_mr.json` is a seed example (Connors' Double 7s mean reversion on ES/NQ).

> _Verified against current code via CodeGraph on 2026-06-07._
