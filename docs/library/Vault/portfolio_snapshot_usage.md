# Portfolio Prediction Materialization Usage

> [!summary]
> Use this flow when you want to fit a `GlobalPortfolio` from working-vault ensembles, materialize its predictions (and per-base-model predictions) into the central cache for research or live workflows, and later reload those rows by `portfolio_id`.

Related: [[Vault/architecture]], [[Vault/user_guide]], [[Vault/portfolio_snapshots_and_predictions]], [[Vault/vault]], [[Cache/architecture]], [[Ensemble/portfolio]].

Paths below use the prop tree (`vault/...`) as examples; pass `vault_root=` / use `vault_personal/...` for the personal vault ([[Vault/vault]]).

> [!warning] No immutable vault snapshot today
> The current code has **no** `GlobalPortfolio.save_to_vault(...)`, **no** `load_global_portfolio_snapshot(...)`, and **no** `<vault_root>/portfolio_snapshots/` directory. Persistence happens through the materialized prediction store under the central cache, keyed by a `portfolio_id` you supply. To "reload" a portfolio you re-assemble it from its working-vault ensemble directories and re-fit, or you read the materialized parquet rows by `portfolio_id`. See [[Vault/portfolio_snapshots_and_predictions]].

## What gets persisted

When you call `materialize_global_portfolio_predictions(...)`:

- the portfolio-level prediction frame is written to `.cache/trading_algo/central_cache/materialized/<scope>/portfolio/<portfolio_id>.parquet`
- per-base-model prediction frames (for identities active in the working vault) are written under `.cache/trading_algo/central_cache/materialized/<scope>/base_models/<TF>/<ensemble_name>/<identity_hash>.parquet`
- rows carry `world` (`train`/`val`/`test`/`live`) and an optional `research_run_id`

Nothing is copied into the vault; the vault remains the source of the *ensemble definitions* the portfolio was built from.

## Typical research workflow

```python
from datetime import datetime

from ensemble.portfolio import (
    GlobalPortfolio,
    PortfolioCacheQuery,
    PortfolioWorld,
    TFPortfolio,
    build_global_portfolio_from_ensemble_dirs,
    discover_ensemble_dirs_in_vault,
    materialize_global_portfolio_predictions,
)
from ensemble.vault_manager import ensure_vault_cache_coverage
from utils.cache import ArtifactScope
from utils.core.enums import TimeFrame

ensemble_dirs = ("vault/M/buy_hold/buy_hold_long",)

# Preflight: make sure bias/EWSD artifacts cover the full train→test window.
ensure_vault_cache_coverage(
    vault_ensemble_dirs=ensemble_dirs,
    start_date=datetime(2010, 1, 1),
    end_date=datetime(2024, 12, 31),
)

# Assemble a GlobalPortfolio from the working-vault ensemble directories.
global_portfolio = build_global_portfolio_from_ensemble_dirs(
    ensemble_dirs,
    max_position_pct=3.5,
)

train_query = PortfolioCacheQuery(
    tickers=("ES",),
    start=datetime(2010, 1, 1),
    end=datetime(2020, 12, 31),
    timeframes=(TimeFrame.M,),
)
test_query = PortfolioCacheQuery(
    tickers=("ES",),
    start=datetime(2021, 1, 1),
    end=datetime(2024, 12, 31),
    timeframes=(TimeFrame.M,),
)

global_portfolio.fit_from_cache(train_query, instrument_returns=instrument_returns_df)

portfolio_id = "buy_hold_es_2024"  # caller-chosen stable id

materialize_global_portfolio_predictions(
    portfolio=global_portfolio,
    query=train_query,
    portfolio_id=portfolio_id,
    world=PortfolioWorld.TRAIN,
    research_run_id="wf_2026_03",
    scope=ArtifactScope.LIVE,
)
materialize_global_portfolio_predictions(
    portfolio=global_portfolio,
    query=test_query,
    portfolio_id=portfolio_id,
    world=PortfolioWorld.TEST,
    research_run_id="wf_2026_03",
    scope=ArtifactScope.LIVE,
)

# "Reload" = read the materialized rows, or re-run predict_from_cache directly.
positions = global_portfolio.predict_from_cache(test_query)
```

`fit_from_cache` and `predict_from_cache` resolve instrument volatility from cached EWSD artifacts. Use `PortfolioWorld.VAL` for validation materialization; the portfolio research pipeline maps its `Validation` phase to `val` automatically.

## Live workflow

For live inference, the pattern is the same:

1. Fit or assemble the deployed `GlobalPortfolio` from working-vault ensembles.
2. Choose a stable `portfolio_id`.
3. Register that `portfolio_id` in `deployment/config/live_cache_refresh.json` so LIVE candle writes keep it materialized automatically.
4. Materialize live predictions with `world=PortfolioWorld.LIVE`.
5. Use `portfolio_id` as the stable join key for monitoring, dashboards, and audits.

Example:

```python
materialize_global_portfolio_predictions(
    portfolio=global_portfolio,
    query=live_query,
    portfolio_id=portfolio_id,
    world=PortfolioWorld.LIVE,
    scope=ArtifactScope.LIVE,
)
```

## Directory quick reference

Materialized outputs:

```text
.cache/trading_algo/central_cache/materialized/<scope>/portfolio/<portfolio_id>.parquet
.cache/trading_algo/central_cache/materialized/<scope>/base_models/<timeframe>/<ensemble_name>/<identity_hash>.parquet
```

`<identity_hash>` is a SHA-256 hash of `(timeframe, ensemble_name, feature_name, model_id)`.

## Cleanup behavior

Use `prune_inactive_base_model_materializations(...)` when you want the base-model materialization tree to match the current working vault.

```python
from ensemble.portfolio import prune_inactive_base_model_materializations
from utils.cache import ArtifactScope

cleanup = prune_inactive_base_model_materializations(
    vault_root="vault",
    scope=ArtifactScope.LIVE,
)
```

Rules:

- active base-model identities are derived from the current working vault
- stale base-model parquet files are deleted
- historical portfolio-level parquet files are kept

## Operational rules

- `fit()` / `fit_from_cache()` do not write anything to disk; materialization is a separate explicit step
- `portfolio_id` is caller-supplied; it is the stable join key for materialized rows
- use `PortfolioWorld.VAL` for validation materialization
- pass `research_run_id` when you want to group rows from the same research run
- rerunning materialization for the same row key updates rows instead of duplicating them

## Common mistakes

- Do not call `GlobalPortfolio.save_to_vault(...)` or `load_global_portfolio_snapshot(...)`; they do not exist. Re-assemble from working-vault ensembles, or read the materialized rows by `portfolio_id`.
- Do not use raw strings like `"validation"` for `world`; use `PortfolioWorld.VAL` or the exact stored value `val`.
- Do not expect cleanup to delete historical portfolio parquet files. Cleanup only targets stale base-model materializations.

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
