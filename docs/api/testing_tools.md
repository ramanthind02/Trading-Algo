# testing_tools

> **Path:** `utils/evaluation/permutation_test/`, `utils/evaluation/robustness_test/`, `utils/simulation/prop_firm_simulator/`, `prop_firms/`, `plotting/`  
> **Status:** Draft  
> **Last updated:** 2026-02-13

## Purpose
This document covers the research/testing APIs for significance testing, resampling-based robustness checks, legacy and provider-oriented prop-firm simulation, plus plotting helpers used by demo/research-style scripts.

## Public API policy (what we document)
This doc includes:
- Re-exports in `utils/evaluation/permutation_test/__init__.py`, `utils/evaluation/robustness_test/__init__.py`, `utils/simulation/prop_firm_simulator/__init__.py`, and `prop_firms/__init__.py`
- Public top-level functions/classes in `plotting/robustness.py` and `plotting/prop_firm.py`
- Public methods required to call those classes effectively

Skipped unless needed for understanding:
- `_`-prefixed helpers and internal implementation details
- Script-local helper functions in `scripts/demo_*.py`

## Quickstart (minimal)
```python
import numpy as np
import pandas as pd

from utils.core.enums import ResamplingMethod
from utils.evaluation.robustness_test import robustness_test

idx = pd.date_range("2024-01-01", periods=252, freq="B")
returns = pd.Series(np.random.default_rng(7).normal(0.0005, 0.01, 252), index=idx)

fig, resampled, stats = robustness_test(
    returns=returns,
    n_samples=200,
    method=ResamplingMethod.BLOCK_BOOTSTRAP,
    block_size=16,
    random_seed=42,
    verbose=False,
)
print(stats["p_value"])
```

```python
from utils.simulation.prop_firm_simulator import (
    PropFirmChallengeSimulator,
    SimulationConfig,
    SimulationMethod,
    ChallengeRules,
    ChallengeCosts,
)

cfg = SimulationConfig(
    method=SimulationMethod.MONTE_CARLO_BLOCK,
    rules=ChallengeRules(max_drawdown_pct=0.10, profit_target_pct=0.08, min_trading_days=5),
    costs=ChallengeCosts(reset_fee=100.0, one_time_fee=500.0),
    n_simulations=200,
    block_length=16,
    random_seed=42,
)
sim = PropFirmChallengeSimulator(returns, cfg)
results, stats = sim.run_simulation(verbose=False)
print(stats.pass_probability)
```

## Data contracts
- **Return series contract** (`robustness_test`, `PropFirmChallengeSimulator`, path generators)
  - type: `pd.Series`
  - index: `pd.DatetimeIndex`
  - values: numeric returns in decimal units (for example `0.01` = 1%)
  - NaN: robustness test drops NaN with warning; prop firm simulator rejects NaN
- **Permutation bar input contract** (`CandleShuffler`, `BarPermute`, `BarPermutationStrategy`)
  - `pd.DataFrame` with `open, high, low, close, datetime`
  - datetime sorted internally (`kind="stable"`) and preserved in output
- **Permutation engine criterion contract**
  - callable signature: `criterion_func(data, feature_col) -> float`
  - expected behavior: larger value = better; p-value counts `permuted >= original`
- **Prop firm result types**
  - `ChallengeResult` is `Union[ChallengePass, ChallengeFail]`
  - both include `total_cost`, `n_resets`, and `equity_curve`; `passed` property is discriminant
- **Plotting contract**
  - robustness plotting expects stats keys created by `robustness_test` (`p_value`, percentiles, etc.)
  - prop firm plotting expects `SimulationStatistics` plus lists of `ChallengeResult`/`ChallengePass`

## Public API reference

### `utils.evaluation.permutation_test`

`PermutationStrategy`  
Type: abstract class  
Signature:
```python
class PermutationStrategy(ABC):
    def permute(self, data: Any, random_seed: int, **kwargs) -> Any: ...
    def validate_data(self, data: Any, **kwargs) -> None: ...
```
Description: Strategy interface used by `PermutationEngine` for feature/bar permutation backends.

`FeaturePermutationStrategy`  
Type: class  
Signature:
```python
class FeaturePermutationStrategy(PermutationStrategy):
    def validate_data(self, data: pd.DataFrame, feature_cols: list[str], target_col: str, **kwargs) -> None: ...
    def permute(self, data: pd.DataFrame, random_seed: int, feature_cols: list[str], exclude_features: list[str] | None = None, train_windows: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None, **kwargs) -> pd.DataFrame: ...
```
Behavior: Shuffles feature columns globally or within each provided window (for walk-forward contamination control).

`BarPermutationStrategy`  
Type: class  
Signature:
```python
class BarPermutationStrategy(PermutationStrategy):
    def validate_data(self, data: Any, bar_data: dict[str, Any] | None = None, **kwargs) -> None: ...
    def permute(self, data: Any, random_seed: int, permute_start_idx: int = 252, bar_data: dict[str, Any] | None = None, train_windows: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None, shuffle_mode: str = "auto", intraday_gap_config: Any | None = None, **kwargs) -> pd.DataFrame: ...
```
Behavior: Shuffles OHLC bars, re-extracts features, and returns a merged feature+price frame.

`PermutationEngine`  
Type: class  
Signature:
```python
class PermutationEngine:
    def __init__(self, strategy: PermutationStrategy, n_jobs: int = -1, verbose: bool = True): ...
    def run_permutation_test(self, data: Any, feature_cols: list[str], criterion_func: Callable, nreps: int = 100, random_seed: int | None = None, alpha: float = 0.05, **strategy_kwargs) -> pd.DataFrame: ...
```
Behavior: Computes original criterion per feature, runs `nreps-1` permutations (parallel when configured), and returns `feature/original_criterion/pval/significant`.

`run_permutation_test`  
Type: function  
Signature:
```python
def run_permutation_test(data: Any, feature_cols: list[str], criterion_func: Callable, strategy: str = "feature", nreps: int = 100, n_jobs: int = -1, random_seed: int | None = None, alpha: float = 0.05, verbose: bool = True, **strategy_kwargs) -> pd.DataFrame: ...
```
Behavior: Convenience wrapper creating the engine and strategy (`"feature"` or `"bar"`).

`CandleShuffleMode`, `GapType`, `IntradayGapConfig`  
Type: enums/dataclass  
Signature:
```python
class CandleShuffleMode(Enum): AUTO, DAILY, INTRADAY
class GapType(Enum): WEEKEND, MAINTENANCE, REGULAR
@dataclass(frozen=True)
class IntradayGapConfig: ...
```
Behavior: Control objects for candle-level shuffling and intraday gap classification.

`classify_daily_gap`, `classify_intraday_gap`  
Type: functions  
Signature:
```python
def classify_daily_gap(prev_ts: pd.Timestamp, curr_ts: pd.Timestamp) -> GapType: ...
def classify_intraday_gap(prev_ts: pd.Timestamp, curr_ts: pd.Timestamp, config: IntradayGapConfig | None = None) -> GapType: ...
```
Behavior: Classify transition type used to keep weekend/maintenance gaps in separate shuffle pools.

`CandleShuffler`  
Type: class  
Signature:
```python
class CandleShuffler:
    def __init__(self, df: pd.DataFrame, permute_start_idx: int = 0, mode: str | CandleShuffleMode = CandleShuffleMode.AUTO, intraday_gap_config: IntradayGapConfig | None = None, random_seed: int | None = None): ...
    def permute(self) -> pd.DataFrame: ...
```
Behavior: Produces permuted OHLC while preserving datetime order and continuity constraints.

`BarPermute`, `BarPermuteWalkForward`, `WalkForwardValidator`  
Type: classes  
Signatures:
```python
class BarPermute:
    def __init__(self, df: pd.DataFrame, permute_start_idx: int = 0, shuffle_mode: str | CandleShuffleMode = CandleShuffleMode.AUTO, intraday_gap_config: IntradayGapConfig | None = None, random_seed: int | None = None): ...
    def permute(self) -> pd.DataFrame: ...

class BarPermuteWalkForward:
    def __init__(self, df: pd.DataFrame, train_windows: list[tuple[pd.Timestamp, pd.Timestamp]]): ...
    def permute(self) -> pd.DataFrame: ...

class WalkForwardValidator:
    def __init__(self, train_start: datetime, train_end: datetime, test_step: int, num_steps: int, floor: float = 0.1, min_test_samples: int = 10, verbose: bool = False): ...
    def validate_single_feature(self, df: pd.DataFrame, feature_col: str, target_col: str, force_steps: list[int] | None = None) -> dict: ...
```
Behavior: Backward-compatible bar shuffle, walk-forward-safe window shuffling, and OOS aggregation utility used in walk-forward permutation flows.

Examples
```python
from utils.evaluation.permutation_test import run_permutation_test

def mean_signal_edge(df, feature):
    return float((df[feature] * df["target"]).mean())

out = run_permutation_test(
    data=my_df,
    feature_cols=["rsi_5", "mom_20"],
    criterion_func=mean_signal_edge,
    strategy="feature",
    nreps=200,
    random_seed=11,
    target_col="target",
)
```

### `utils.evaluation.robustness_test`

`ResamplingStrategy`, `MonteCarloStrategy`, `BootstrapStrategy`, `BlockBootstrapStrategy`  
Type: abstract class + classes  
Signatures:
```python
class ResamplingStrategy(ABC):
    def resample(self, returns: pd.Series, random_seed: int, **kwargs) -> pd.Series: ...
    def validate_data(self, returns: pd.Series, **kwargs) -> None: ...

class MonteCarloStrategy(ResamplingStrategy): ...
class BootstrapStrategy(ResamplingStrategy): ...
class BlockBootstrapStrategy(ResamplingStrategy):
    def validate_data(self, returns: pd.Series, block_size: int | None = None, **kwargs) -> None: ...
    def resample(self, returns: pd.Series, random_seed: int, block_size: int, **kwargs) -> pd.Series: ...
```
Behavior: Resample returns while preserving index shape; block bootstrap requires `block_size` and partially preserves autocorrelation.

`robustness_test`  
Type: function  
Signature:
```python
def robustness_test(returns: pd.Series, n_samples: int = 1000, method: ResamplingMethod = ResamplingMethod.MONTE_CARLO, block_size: int | None = None, random_seed: int | None = None, figsize: tuple[int, int] = (14, 8), show_original: bool = True, alpha: float = 0.05, save_path: str | None = None, verbose: bool = True) -> tuple[plt.Figure, list[pd.Series], dict[str, Any]]: ...
```
Behavior: Runs resampling, computes cumulative-return significance statistics, and returns figure + raw resampled return paths + stats dict.

Notes / Constraints
- Monte Carlo and bootstrap destroy temporal order; block bootstrap preserves local structure only.
- P-value is one-sided and conditioned on sign of original cumulative return.
- Seeds use `base_seed + iteration`; default base seed is `42` when not supplied.

### `utils.simulation.prop_firm_simulator`

`SimulationMethod`  
Type: enum  
Signature: `class SimulationMethod(Enum): HISTORICAL_WALKFORWARD, MONTE_CARLO_BLOCK`  
Behavior: Selects path-generation mode.

`ChallengeRules`, `ChallengeCosts`, `SimulationConfig`  
Type: frozen dataclasses  
Signatures:
```python
@dataclass(frozen=True)
class ChallengeRules: ...

@dataclass(frozen=True)
class ChallengeCosts:
    def compute_initial_cost(self) -> float: ...

@dataclass(frozen=True)
class SimulationConfig:
    def get_path_length(self) -> int: ...
```
Behavior: Validate rule/cost/simulation settings; config enforces `block_length` for Monte Carlo block mode.

`ChallengePass`, `ChallengeFail`, `ChallengeResult`, `SimulationStatistics`  
Type: dataclasses + union alias  
Signatures:
```python
@dataclass
class ChallengePass:
    @property
    def passed(self) -> bool: ...

@dataclass
class ChallengeFail:
    @property
    def passed(self) -> bool: ...

ChallengeResult = Union[ChallengePass, ChallengeFail]

@dataclass
class SimulationStatistics: ...
```
Behavior: Strongly typed simulation output objects used by statistics and plotting.

`PathGenerator`, `HistoricalWalkForwardGenerator`, `MonteCarloBlockGenerator`  
Type: abstract class + classes  
Signature:
```python
class PathGenerator(ABC):
    def generate(self, returns: pd.Series, iteration: int, random_seed: int, path_length: int | None = None) -> pd.Series: ...

class HistoricalWalkForwardGenerator(PathGenerator): ...
class MonteCarloBlockGenerator(PathGenerator):
    @property
    def block_length(self) -> int: ...
```
Behavior: Generate historical or synthetic return paths; historical method uses random start and business-day index regeneration.

`PropFirmChallengeSimulator`  
Type: class  
Signature:
```python
class PropFirmChallengeSimulator:
    def __init__(self, returns: pd.Series, config: SimulationConfig): ...
    def run_simulation(self, verbose: bool = False) -> tuple[list[ChallengeResult], SimulationStatistics]: ...
    @property
    def config(self) -> SimulationConfig: ...
    @property
    def returns(self) -> pd.Series: ...
```
Behavior: Runs multi-attempt challenge simulations with reset fees, drawdown checks, and pass/fail classification.

`compute_statistics`, `print_statistics`  
Type: functions  
Signature:
```python
def compute_statistics(results: list[ChallengeResult], config: SimulationConfig) -> SimulationStatistics: ...
def print_statistics(stats: SimulationStatistics, verbose: bool = True) -> None: ...
```
Behavior: Aggregate and print simulation outcomes (pass rate, cost, resets, failure reasons, day distributions).

Examples
```python
from utils.simulation.prop_firm_simulator import print_statistics
print_statistics(stats, verbose=True)
```

### `prop_firms`

`SimulationRequest`, `SimulationResult`, `SimulationSummary`  
Type: dataclasses  
Behavior: Shared provider-oriented request/result contracts for daily return driven prop-firm simulations.

`PayoutPolicy`, `ResetPolicy`, `AccountPhase`, `AccountStatus`, `BreachReason`  
Type: enums  
Behavior: Typed lifecycle controls and terminal status values used by provider engines.

`LucidRuleEngine`, `create_lucid_simulator`, `load_lucid_account`, `load_lucid_accounts`  
Type: class + functions  
Signature:
```python
class LucidRuleEngine(BasePropFirmEngine[LucidState]):
    def simulate(self, request: SimulationRequest) -> SimulationResult: ...

def create_lucid_simulator(config_path: Path | None = None) -> LucidRuleEngine: ...
def load_lucid_account(account_code: str, config_path: Path | None = None) -> AccountDefinition: ...
def load_lucid_accounts(config_path: Path | None = None) -> dict[str, AccountDefinition]: ...
```
Behavior: Loads LucidFlex account definitions from `prop_firms/lucid/config.json` and simulates evaluation, funded, scaling-plan, payout, and optional holdings/exposure validation through one daily engine.

Examples
```python
import pandas as pd

from prop_firms import SimulationRequest, create_lucid_simulator

simulator = create_lucid_simulator()
result = simulator.simulate(
    SimulationRequest(
        account_code="25000",
        returns=pd.Series(
            [0.026, 0.024, 0.01],
            index=pd.date_range("2026-01-05", periods=3, freq="B"),
        ),
    )
)
print(result.summary.final_status)
```

### `plotting` APIs used by scripts

`plotting.robustness.plot_robustness_curves`  
Type: function  
Signature:
```python
def plot_robustness_curves(original_curve: pd.Series, resampled_curves: list[pd.Series], stats: dict[str, Any], figsize: tuple[int, int] = (14, 8), show_original: bool = True, alpha: float = 0.05, method: ResamplingMethod = ResamplingMethod.MONTE_CARLO, save_path: str | None = None) -> plt.Figure: ...
```
Behavior: Plots all paths, percentile bands, median path, and optional original path colored by significance.

`plotting.prop_firm.plot_challenge_results`  
Type: function  
Signature:
```python
def plot_challenge_results(results: list[ChallengeResult], stats: SimulationStatistics, config: SimulationConfig, figsize: tuple[int, int] = (14, 10), save_path: str | None = None) -> plt.Figure: ...
```
Behavior: 2x2 dashboard (pass/fail pie, days histogram, sample curves, cost histogram).

`plotting.prop_firm.plot_equity_curves_sample`  
Type: function  
Signature:
```python
def plot_equity_curves_sample(results: list[ChallengeResult], config: SimulationConfig, n_samples: int = 20, figsize: tuple[int, int] = (12, 6), save_path: str | None = None) -> plt.Figure: ...
```
Behavior: Percentile band + median + sampled equity curves.

`plotting.prop_firm.plot_days_to_pass_distribution`  
Type: function  
Signature:
```python
def plot_days_to_pass_distribution(passes: list[ChallengePass], stats: SimulationStatistics, figsize: tuple[int, int] = (10, 6), save_path: str | None = None) -> plt.Figure: ...
```
Behavior: Density-style histogram with distribution statistic markers.

`plotting.prop_firm.plot_cost_analysis`  
Type: function  
Signature:
```python
def plot_cost_analysis(results: list[ChallengeResult], stats: SimulationStatistics, figsize: tuple[int, int] = (10, 6), save_path: str | None = None) -> plt.Figure: ...
```
Behavior: Cost histogram + reset-count histogram with expected-value markers.

## Internal but required
- `BarPermutationStrategy` relies on `utils.evaluation.permutation_test.permute_bars._extract_features_from_bars(...)`; this is currently a compatibility placeholder returning requested feature columns and ATR defaults.
- `PropFirmChallengeSimulator` uses `metrics.risk.drawdown.check_drawdown_breach(...)` for rule enforcement; challenge pass/fail semantics depend on that function's reason labels.

## Errors & logging
- Common exceptions:
  - `TypeError`/`ValueError` from input validators (bad series type/index, missing columns, invalid config values, missing `block_size`/`bar_data`)
  - `ValueError("Unknown strategy")` for invalid permutation strategy strings
  - `ValueError("Unknown resampling method")` for unsupported resampling enum values
- Logging/output behavior:
  - These modules primarily use `print` for progress and summaries (`verbose=True`), not structured logger events
  - Plotting functions print save location when `save_path` is provided
  - Robustness test prints a warning before dropping NaN returns

## Randomization & reproducibility
- Permutation and robustness engines derive per-iteration seeds deterministically as `base_seed + iteration`.
- `random_seed=None` paths are not fully reproducible:
  - `PermutationEngine` leaves permutation RNG unset (global/random state behavior)
  - `robustness_test` and `PropFirmChallengeSimulator` default to base seed `42` and are reproducible by default
  - `CandleShuffler` without `random_seed` uses global `np.random.permutation`
- Parallel permutation (`n_jobs != 1`) should remain deterministic for a fixed seed and process count because each replication gets a fixed derived seed.

## Open questions
Q1: Should deterministic behavior for `PermutationEngine` with `random_seed=None` be standardized (for example by adopting an explicit default base seed) to match robustness/simulator defaults?

Q2: Should the compatibility placeholder in `utils.evaluation.permutation_test.permute_bars._extract_features_from_bars(...)` remain public-facing, or be replaced with a required feature-extraction callback contract?
