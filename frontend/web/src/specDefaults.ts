import type { StrategySpec } from "./api/types";

/** A blank, valid-shaped spec for the "New strategy" flow. */
export function blankSpec(): StrategySpec {
  return {
    spec_version: "1.0",
    name: "",
    hypothesis: "",
    author: "",
    created: new Date().toISOString(),
    tickers: [],
    mode: "daily",
    timeframe: "D",
    windows: null,
    signal: { module_name: "", param_grid: {} },
    direction: "long_short",
    vol_scaling: "blended",
    vol_scaling_model: "inherit_daily",
    risk: { target_vol: 0.15, forecast_cap: 2.0, max_position_pct: 3.5, buffer_fraction: 0.0 },
    account: { capital: 50000, prop_constraints: null },
    execution: {
      entry_policy: "market_on_open",
      exit_policy: "market_on_open",
      unfilled_limit: "cross_after",
      holding: "overnight",
      fill_feed: "derived",
    },
    vault: { weight_hierarchy_group: "mean_reversion_indices", ensemble_name: "" },
  };
}
