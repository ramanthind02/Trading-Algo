// TypeScript mirrors of the StrategySpec JSON (research/spec/serialization.py) and the API
// responses (frontend/api). Kept hand-written and small; the spec shape is the contract.

export type ParamValue = number | string | boolean;

export interface WindowBound {
  start: string; // ISO datetime
  end: string;
}

export interface SpecWindows {
  train: WindowBound;
  validation: WindowBound;
  test: WindowBound;
}

export interface SignalSpec {
  module_name: string;
  param_grid: Record<string, ParamValue[]>;
}

export interface RiskSpec {
  target_vol: number;
  forecast_cap: number;
  max_position_pct: number;
  buffer_fraction: number;
}

export interface PropConstraints {
  max_daily_loss_pct: number | null;
  max_total_loss_pct: number | null;
  profit_target_pct: number | null;
  min_trading_days: number | null;
  max_leverage: number | null;
  news_trading_restricted: boolean;
}

export interface AccountSpec {
  capital: number;
  prop_constraints: PropConstraints | null;
}

export interface ExecutionSpec {
  entry_policy: string;
  exit_policy: string;
  unfilled_limit: string;
  holding: string;
  fill_feed: string;
}

export interface VaultTarget {
  weight_hierarchy_group: string;
  ensemble_name: string;
}

export interface StrategySpec {
  spec_version: string;
  name: string;
  hypothesis: string;
  author: string;
  created: string;
  tickers: string[];
  /** Legacy/optional. Signals run on the futures-additive feed; results fan out to both lanes. */
  data_feed?: string;
  mode: string;
  timeframe: string;
  windows: SpecWindows | null;
  signal: SignalSpec;
  direction: string;
  vol_scaling: string;
  vol_scaling_model: string;
  risk: RiskSpec;
  account: AccountSpec;
  execution: ExecutionSpec;
  vault: VaultTarget;
}

export interface EnumOption {
  value: string;
  label: string;
}

export interface FormSchema {
  tickers: string[];
  timeframes: { daily: string[]; intraday: string[]; all: string[] };
  modes: EnumOption[];
  directions: EnumOption[];
  vol_scalings: EnumOption[];
  vol_scaling_models: EnumOption[];
  order_policies: EnumOption[];
  unfilled_limit_policies: EnumOption[];
  holdings: EnumOption[];
  fill_feeds: EnumOption[];
  vault_sleeves: string[];
  max_grid_combos: number;
}

export interface ModuleCatalogItem {
  name: string;
  class_name: string;
  category: string;
  import_path: string;
}

export interface ModuleParam {
  name: string;
  required: boolean;
  default: ParamValue | null;
  annotation: string | null;
  choices?: string[];
}

export interface ModuleDetail {
  name: string;
  class_name: string;
  import_path: string;
  doc: string;
  params: ModuleParam[];
}

export interface SpecValidationDerived {
  num_combos: number;
  resolved_fill_feed: string;
  uses_limit: boolean;
  feed_literal: string;
  windows_explicit: boolean;
  windows: { train: [string, string]; validation: [string, string]; test: [string, string] } | null;
}

export interface SpecValidation {
  valid: boolean;
  error: string | null;
  derived: SpecValidationDerived | null;
}

export interface SpecSummary {
  id: string;
  name: string;
  hypothesis: string;
  module: string | null;
  tickers: string[];
  timeframe: string | null;
  mode: string | null;
  direction: string | null;
  data_feed: string | null;
  vault_sleeve: string | null;
  num_combos: number | null;
  valid: boolean;
  error: string | null;
  updated_at: string | null;
}

export interface SpecDetail {
  id: string;
  spec: StrategySpec;
  validation: SpecValidation;
}

export type RunStatus = "queued" | "running" | "completed" | "failed";

export interface SpecRun {
  run_id: string;
  spec_id: string;
  spec_name: string;
  phase: string;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  num_combos: number;
  reports_dir: string;
  viz_dir: string;
  started_at: string | null;
  finished_at: string | null;
  log_text: string;
  error_text: string | null;
  exit_code: number | null;
}

export interface ArtifactEntry {
  name: string;
  path: string;
  kind: "csv" | "json" | "markdown" | "text" | "image" | "html" | "binary";
  size: number;
  mtime: number;
}

export interface ArtifactGroup {
  label: string;
  root: string;
  files: ArtifactEntry[];
}

export interface CsvPreview {
  kind: "csv";
  path: string;
  columns: string[];
  rows: Record<string, ParamValue | null>[];
  total_rows: number;
  truncated: boolean;
}

export interface JsonPreview {
  kind: "json";
  path: string;
  data: unknown;
}

export interface TextPreview {
  kind: "markdown" | "text";
  path: string;
  text: string;
}

export interface MediaPreview {
  kind: "image" | "html";
  path: string;
  raw_url: string;
}

export type ArtifactPreview = CsvPreview | JsonPreview | TextPreview | MediaPreview | { kind: "binary"; path: string };

// --- results dashboard ---

export interface GridRow {
  label: string;
  params: Record<string, number | string>;
  n_observations: number | null;
  n_nonzero_signal: number | null;
  sharpe: number | null;
  t_stat: number | null;
  sortino: number | null;
  nw_sharpe: number | null;
  selection_score: number | null;
  is_best: boolean;
}

export interface PlateauPoint {
  label: string;
  params: Record<string, number | string>;
  metrics: Record<string, number | null>;
  is_best: boolean;
}

export interface EquitySeries {
  combo: string;
  ticker: string;
  datetime: string[];
  cumulative: number[];
}

export interface ValidationImage {
  label: string;
  path: string; // repo-relative; served via /api/artifacts/raw?path=...
}

export interface ValidationSummary {
  sr_is?: number | null;
  sr_val?: number | null;
  degradation_ratio?: number | null;
  degradation_z?: number | null;
  ci_overlap?: boolean | null;
  robustness_passed?: boolean | null;
  robustness_interpretation?: string | null;
  cusum_break?: boolean | null;
  gate_passed?: boolean | null;
  gate_interpretation?: string | null;
  gate_n_existing?: number | null;
  gate_weight_method?: string | null;
  gate_mean_peer_corr?: number | null;
}

/** One result lane (futures / cfd / default) — the old flat results block. */
export interface LaneResults {
  headline: Record<string, string | number | boolean | null> | null;
  grid: { available: boolean; rows: GridRow[]; param_keys: string[] };
  plateau: {
    available: boolean;
    swept_params?: string[];
    metric_options?: string[];
    points?: PlateauPoint[];
  };
  equity: { available: boolean; series: EquitySeries[]; combos: string[]; tickers: string[] };
  // Validation-only fields (absent on exploration lanes):
  validation_summary?: ValidationSummary;
  validation_images?: ValidationImage[];
}

/**
 * The /results response. Exploration runs fan out to `futures` + `cfd`; validation/legacy
 * runs return a single `default` lane. Always read out.lanes[out.primary_lane].
 */
export interface RunResultsResponse {
  lanes: Record<string, LaneResults>;
  lane_order: string[];
  primary_lane: string;
}

// --- conditional returns (binning) ---

export interface ConditionalBin {
  bin_index: number;
  label: string;
  lo: number | null;
  hi: number | null;
  regime: "on" | "off" | null;
  ticker: string | null;
  n_obs: number;
  mean_return: number | null;
  sharpe: number | null;
  sortino: number | null;
  t_stat: number | null;
  hit_rate: number | null;
  cumulative: number | null;
  instrument_mean_return: number | null;
}

export interface ConditionalReturnsData {
  available: boolean;
  reason: string | null;
  bin_mode: "quantile" | "fixed";
  n_bins: number;
  per_ticker: boolean;
  tickers: string[];
  timeframe: string;
  strategy: { module: string; params: Record<string, ParamValue>; label: string; direction: string };
  condition: { module: string; label: string };
  regime: { label: string } | null;
  metric_options: string[];
  bins: ConditionalBin[];
}

export interface IndicatorChoice {
  module: string;
  params: Record<string, ParamValue>;
}

export interface ConditionalReturnsRequest {
  indicator: IndicatorChoice;
  bin_mode: "quantile" | "fixed";
  n_bins: number;
  edges: number[];
  regime: (IndicatorChoice & { threshold: number; above: boolean }) | null;
  per_ticker: boolean;
}

// --- vault promotion ---

export interface VaultEligibility {
  configured: boolean;
  ready: boolean;
  gate_status: string;
  blockers: string[];
  target: Record<string, string | null> | null;
  module_name?: string;
  cli_command?: string;
}

export interface VaultSaveResult {
  dry_run: boolean;
  vault_root: string;
  ensemble_dir: string;
  ensemble_dir_repo_relative: string;
  tickers: string[];
  direction: string;
  feature_column: string;
  model_id: string | null;
  vault_profile: string;
  weight_hierarchy_group: string | null;
}

export interface VaultPreview {
  eligibility: VaultEligibility;
  preview: VaultSaveResult | null;
  preview_error: string | null;
}

// --- portfolio research ---

export interface PortfolioPhaseOption {
  value: string;
  label: string;
  description?: string;
}

export interface PortfolioDefaults {
  default_phase: string;
  default_tickers?: string;
  ticker_options?: string[];
  phase_options: PortfolioPhaseOption[];
  ensemble_count: number;
  portfolio_fit_mode: string;
  weight_layer_policy: string;
  config_source_path?: string;
  output_root: string;
  // editable config (current values + option lists)
  tickers: string[];
  fit_mode: string;
  fit_modes: string[];
  weight_layer_method: string;
  weight_layer_methods: string[];
  target_volatility: number;
  max_position_pct: number;
}

export interface PortfolioRunOverrides {
  phase?: string;
  tickers?: string[];
  fit_mode?: string;
  weight_layer_method?: string;
  target_volatility?: number;
  max_position_pct?: number;
  // SR-tilt knobs + an optional hierarchy-structure override
  sr_adjustment?: boolean;
  sr_tilt_max_depth?: number;
  within_group_method?: string;
  sr_avg?: number;
  fdm_max?: number;
  hierarchy_spec?: HierarchyNode;
}

// --- weight layer ---

export interface HierarchyNode {
  type: "group" | "leaf";
  id?: string;
  stream_id?: string;
  children?: HierarchyNode[];
}

export interface WeightRow {
  phase: string;
  path: string; // e.g. root/equity_indices/mean_reversion_indices/<stream>
  weight: number | null;
  fdm: number | null;
  ticker: string | null;
  timeframe: string | null;
  stream_id: string;
  mean_signal_corr: number | null;
}

export interface WeightLayerData {
  method: string;
  policy: string;
  sr: {
    sr_adjustment: boolean;
    sr_avg: number | null;
    sr_p_step: number | null;
    sr_min_years: number | null;
    sr_tilt_max_depth: number | null;
    within_group_method: string;
    fdm_max: number | null;
  };
  within_group_methods: string[];
  hierarchy: HierarchyNode | null;
  weights: { available: boolean; phases: string[]; rows: WeightRow[] };
}

export interface PortfolioJob {
  job_id: string;
  phase: string;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
  plan_title: string;
  output_path: string;
  log_text: string;
  error_text: string | null;
  exit_code: number | null;
}

export interface PortfolioDefaultsResponse {
  defaults: PortfolioDefaults;
  job: PortfolioJob | null;
}

// --- vault features archive ---

export interface VaultFeature {
  profile: string;
  ensemble_name: string;
  timeframe: string | null;
  sleeve: string | null;
  direction: string | null;
  tickers: string[];
  created_at: string | null;
  updated_at: string | null;
  path: string;
  feature_count: number;
}

// --- live monitoring (node-mediated dashboard feed) ---
// Mirrors the snapshot the live node publishes (deployment/live/monitoring/live_state.py)
// and the derived gauges the API computes (frontend/api/live.py).

export interface LiveBrokerSummary {
  broker: string;
  confirmed: boolean;
  has_snapshot: boolean;
  online: boolean;
  age_seconds: number | null;
  exec_tier: string | null;
  halted: boolean | null;
  has_risk_limits: boolean;
}

export interface LiveAccount {
  login: number;
  server: string;
  currency: string;
  balance: number | null;
  floating_pnl: number | null;
  equity: number | null;
  leverage_in_use: number | null;
}

export interface LivePosition {
  canonical: string | null;
  symbol: string;
  instrument_id: string;
  side: "LONG" | "SHORT" | "FLAT";
  net_qty: number;
  avg_px_open: number;
  last_px: number | null;
  mark_age_secs: number | null;
  mark_stale: boolean;
  unrealized_pnl: number | null;
  notional: number | null;
}

export interface LiveTarget {
  canonical: string;
  target_fraction: number;
  forecast_score: number;
  target_qty: number | null;
  current_qty: number;
  drift: number | null;
}

export interface LiveWarmupTicker {
  ticker: string;
  bars: number;
  ready: boolean;
}

export interface LiveWarmup {
  ready: boolean;
  min_bars: number;
  as_of: string | null;
  per_ticker: LiveWarmupTicker[];
}

export interface LiveCommandResult {
  id: string;
  action: string;
  status: string;
  detail: string;
  at: string;
}

export interface LiveRiskBaseline {
  account_start_equity: number | null;
  account_start_at: string | null;
  day_start_equity: number | null;
  day_start_balance: number | null;
  day_start_date: string | null;
  day_start_estimated: boolean;
}

export interface LiveSnapshot {
  schema_version: number;
  broker: string;
  venue: string;
  exec_tier: string;
  account: LiveAccount;
  positions: LivePosition[];
  gross_notional: number;
  marks_fresh: boolean;
  targets: LiveTarget[];
  warmup: LiveWarmup | null;
  strategy_state: string;
  halted: boolean;
  ready_to_trade: boolean;
  initial_balance: number | null;
  risk_baseline: LiveRiskBaseline;
  last_command_result: LiveCommandResult | null;
  node_started_at: string | null;
  ts: string;
  age_seconds: number | null;
  online: boolean;
}

export interface LiveRiskGauges {
  daily_pnl_pct: number | null;
  daily_drawdown_pct: number | null;
  daily_limit_pct: number | null;
  daily_headroom_pct: number | null;
  daily_used_pct: number | null;
  total_pnl_pct: number | null;
  total_drawdown_pct: number | null;
  total_limit_pct: number | null;
  total_headroom_pct: number | null;
  total_used_pct: number | null;
  profit_target_pct: number | null;
  profit_progress_pct: number | null;
  leverage_in_use: number | null;
  max_leverage: number | null;
}

export interface LiveRiskLimits {
  max_daily_loss_pct: number | null;
  max_total_loss_pct: number | null;
  profit_target_pct: number | null;
  min_trading_days: number | null;
  max_leverage: number | null;
  news_trading_restricted: boolean;
}

export interface LiveRisk {
  broker: string;
  has_limits: boolean;
  limits: LiveRiskLimits | null;
  gauges: LiveRiskGauges;
  equity: number | null;
  baseline: LiveRiskBaseline;
  day_start_estimated: boolean;
  online: boolean;
}

export interface LiveEquityPoint {
  ts: string;
  equity: number;
}

export interface LiveEquitySeries {
  broker: string;
  series: LiveEquityPoint[];
}

export interface FlattenResult {
  status: string;
  command_id: string;
  broker: string;
  issued_at: string;
  note: string;
}

// --- data coverage (data registry) ---

export interface DataCoverageStore {
  store: string;
  key_count: number;
  earliest: string | null;
  latest: string | null;
  total_rows: number | null;
}

export interface DataFreshnessStore {
  store: string;
  coverage_end: string | null;
}

export interface DataJobRun {
  job_run_id: number;
  job_name: string;
  args_json: string | null;
  started_at: string;
  finished_at: string | null;
  broker_time_anchor: string | null;
  exit_code: number | null;
  rows_written: number | null;
  coverage_json: string | null;
  error_text: string | null;
}

export interface DataAccount {
  account_id: number;
  broker: string;
  login: string;
  exec_tier: string;
  program_phase: string;
  status: string;
  valid_from: string;
  valid_to: string | null;
  currency: string;
  initial_balance: number | null;
}
