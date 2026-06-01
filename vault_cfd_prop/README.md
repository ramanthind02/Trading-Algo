# Vault CFD Prop - CFD Prop-Firm Working Model Storage

This vault stores validated trading features and saved base-model state for the
**CFD prop-firm** profile (`scripts/enigma_cfd_prop_forecast.py`), which trades
US100 / US500 / XAUUSD / XAGUSD via MetaTrader 5 across one or more prop-firm
accounts (FTMO, MyForexFunds, etc.).

It is **structurally identical** to `vault/` (see `vault/README.md` and
`docs/library/Vault/architecture.md`) and supports the same nested
`<TF>/<weight_hierarchy_group>/<ensemble_leaf>/` layout. The CFD orchestrator
filters ensembles by `tradeable_tickers` declared in the CFD config, so
ensembles for unsupported instruments are simply ignored at forecast time.

## Provenance

This vault was **seeded by copying applicable ensembles from `vault/`** (the
futures prop-firm vault) at the time the CFD prop-firm profile was introduced.
The two vaults are expected to diverge over time as CFD-specific validation
(broker-specific session times, weekend gaps, CFD-only instruments) accumulates.

Currently seeded:

- `D/mean_reversion_indices/` (ES, NQ)
- `D/momentum/`                 (indices momentum)
- `D/momentum_gc/`              (GC)
- `M/buy_hold/`                 (long-bias monthly)

`SI` (XAGUSD) has no ensemble yet — that is expected. The orchestrator skips
tickers without a signal.

## Path resolution

The vault root is resolved by `utils/vault_paths.py::resolve_vault_cfd_prop()`:

- if `TRADING_ALGO_VAULT_CFD_PROP` is set in the environment, that path is used;
- otherwise the default is `<repo>/vault_cfd_prop`.

## Adding new CFD-specific ensembles

Same flow as `vault/`. See:

- `docs/library/Vault/architecture.md`
- `docs/library/Vault/user_guide.md`

To save a research model into this vault specifically, point
`VaultSaveConfig.vault_root` (or set the env var) at `vault_cfd_prop` before
running `python -m feature_research.save_feature_to_vault`.
