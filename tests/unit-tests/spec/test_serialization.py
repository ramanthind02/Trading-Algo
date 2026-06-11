"""Round-trip tests for StrategySpec JSON (de)serialization.

The JSON file is the shared contract between an agent (writes the file) and the frontend Spec
Builder (loads/edits/saves it); these tests assert that ``spec → dict → spec`` and
``spec → json → spec`` are lossless and that a deserialized spec is fully validated.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from research.spec.serialization import (
    load_spec,
    save_spec,
    spec_from_dict,
    spec_from_json,
    spec_to_dict,
    spec_to_json,
)
from research.spec.strategy_spec import (
    AccountSpec,
    DataFeed,
    Direction,
    ExecutionSpec,
    FillFeed,
    Holding,
    OrderPolicy,
    PropConstraints,
    ResearchWindows,
    RiskSpec,
    SignalSpec,
    StrategyMode,
    StrategySpec,
    Ticker,
    TimeFrame,
    UnfilledLimitPolicy,
    VaultTarget,
    VolScaling,
    VolScalingModel,
)


def _minimal_spec(**overrides) -> StrategySpec:
    base = dict(
        name="daily_rsi_mr",
        hypothesis="RSI(2) mean reversion on equity indices.",
        author="tester",
        created=datetime(2026, 1, 1),
        tickers=(Ticker.ES,),
        data_feed=DataFeed.DARWINEX_CFD,
        mode=StrategyMode.DAILY,
        timeframe=TimeFrame.D,
        signal=SignalSpec("rsi_signal", {"rsi_period": [2, 3], "oversold": [20, 25]}),
        direction=Direction.LONG_SHORT,
        vault=VaultTarget("mean_reversion_indices", "rsi_mr"),
    )
    base.update(overrides)
    return StrategySpec(**base)


def _full_spec() -> StrategySpec:
    """A spec exercising every field: explicit windows, prop constraints, limit execution."""

    return StrategySpec(
        name="audnzd_bollinger_mr",
        hypothesis="AUDNZD reverts to its Bollinger mid-band; both currencies are commodity FX.",
        author="raman",
        created=datetime(2026, 6, 7, 12, 30, 0),
        spec_version="1.1",
        tickers=(Ticker.ES, Ticker.NQ),
        data_feed=DataFeed.NORGATE_FUTURES,
        mode=StrategyMode.INTRADAY,
        timeframe=TimeFrame.H1,
        windows=ResearchWindows(
            train=(datetime(2018, 1, 1), datetime(2022, 12, 31)),
            validation=(datetime(2023, 1, 1), datetime(2024, 12, 31)),
            test=(datetime(2025, 1, 1), datetime(2026, 1, 1)),
        ),
        signal=SignalSpec("bollinger_signal", {"period": [20], "num_std": [2.0, 2.5, 3.0]}),
        direction=Direction.SHORT,
        vol_scaling=VolScaling.LONG_ONLY,
        vol_scaling_model=VolScalingModel.INTRADAY_CUSTOM,
        risk=RiskSpec(target_vol=0.2, forecast_cap=1.5, max_position_pct=2.0, buffer_fraction=0.1),
        account=AccountSpec(
            capital=100_000.0,
            prop_constraints=PropConstraints(
                max_daily_loss_pct=5.0,
                max_total_loss_pct=10.0,
                profit_target_pct=8.0,
                min_trading_days=4,
                max_leverage=30,
                news_trading_restricted=True,
            ),
        ),
        execution=ExecutionSpec(
            entry_policy=OrderPolicy.LIMIT_AT_TOUCH,
            exit_policy=OrderPolicy.MARKET_ON_OPEN,
            unfilled_limit=UnfilledLimitPolicy.CARRY,
            holding=Holding.INTRADAY,
            fill_feed=FillFeed.TICKS,
        ),
        vault=VaultTarget("mean_reversion_indices", "audnzd_bollinger_mr"),
    )


def _assert_specs_equal(left: StrategySpec, right: StrategySpec) -> None:
    assert spec_to_dict(left) == spec_to_dict(right)


# ── dict round-trip ─────────────────────────────────────────────────────────


def test_minimal_dict_round_trip():
    spec = _minimal_spec()
    _assert_specs_equal(spec_from_dict(spec_to_dict(spec)), spec)


def test_full_dict_round_trip():
    spec = _full_spec()
    restored = spec_from_dict(spec_to_dict(spec))
    _assert_specs_equal(restored, spec)
    # spot-check the awkward fields survived
    assert restored.windows is not None
    assert restored.windows.train[0] == datetime(2018, 1, 1)
    assert restored.account.prop_constraints is not None
    assert restored.account.prop_constraints.news_trading_restricted is True
    assert restored.execution.resolved_fill_feed() is FillFeed.TICKS
    assert restored.vol_scaling is VolScaling.LONG_ONLY


# ── json round-trip ─────────────────────────────────────────────────────────


def test_full_json_round_trip():
    spec = _full_spec()
    _assert_specs_equal(spec_from_json(spec_to_json(spec)), spec)


def test_json_is_human_readable_tokens():
    payload = spec_to_dict(_full_spec())
    assert payload["timeframe"] == "H1"  # name, not the int value
    assert payload["tickers"] == ["ES", "NQ"]  # names
    assert payload["direction"] == "short"  # value
    assert payload["vol_scaling"] == "long_only"
    assert payload["execution"]["entry_policy"] == "limit_at_touch"
    assert payload["windows"]["train"]["start"] == "2018-01-01T00:00:00"


# ── file round-trip ─────────────────────────────────────────────────────────


def test_save_and_load_file(tmp_path):
    spec = _full_spec()
    path = save_spec(spec, tmp_path / "nested" / "audnzd.json")
    assert path.exists()
    _assert_specs_equal(load_spec(path), spec)


# ── tolerant reads / validation on load ─────────────────────────────────────


def test_enum_accepts_name_or_value():
    payload = spec_to_dict(_minimal_spec())
    payload["timeframe"] = "D"          # name
    payload["data_feed"] = "darwinex_cfd"  # value
    payload["direction"] = "LONG_SHORT"    # name form of a value-serialized enum
    spec = spec_from_dict(payload)
    assert spec.timeframe is TimeFrame.D
    assert spec.data_feed is DataFeed.DARWINEX_CFD
    assert spec.direction is Direction.LONG_SHORT


def test_defaults_filled_when_optional_blocks_omitted():
    payload = spec_to_dict(_minimal_spec())
    del payload["risk"]
    del payload["account"]
    del payload["execution"]
    del payload["vol_scaling"]
    spec = spec_from_dict(payload)
    assert spec.risk == RiskSpec()
    assert spec.account == AccountSpec()
    assert spec.execution == ExecutionSpec()
    assert spec.vol_scaling is VolScaling.BLENDED


def test_missing_required_field_raises():
    payload = spec_to_dict(_minimal_spec())
    del payload["vault"]
    with pytest.raises(ValueError, match="missing required field 'vault'"):
        spec_from_dict(payload)


def test_validation_runs_on_load_grid_cap():
    payload = spec_to_dict(_minimal_spec())
    payload["signal"]["param_grid"] = {"a": list(range(20)), "b": list(range(20))}  # 400 > 300
    with pytest.raises(ValueError, match="exceeding the cap"):
        spec_from_dict(payload)


def test_validation_runs_on_load_mode_timeframe():
    payload = spec_to_dict(_minimal_spec())
    payload["mode"] = "intraday"  # but timeframe is still D
    with pytest.raises(ValueError, match="INTRADAY mode requires"):
        spec_from_dict(payload)


def test_invalid_enum_token_raises():
    # A non-data_feed enum still raises on an unknown token.
    payload = spec_to_dict(_minimal_spec())
    payload["direction"] = "sideways"
    with pytest.raises(ValueError, match="not a valid Direction"):
        spec_from_dict(payload)


def test_data_feed_is_optional_and_tolerant():
    # data_feed is now an internal label only (signals are always additive futures). Reading
    # tolerates it being absent and degrades an unknown/legacy token to the futures default,
    # rather than raising — so legacy/agent payloads always load.
    base = spec_to_dict(_minimal_spec())

    omitted = {k: v for k, v in base.items() if k != "data_feed"}
    assert spec_from_dict(omitted).data_feed is DataFeed.NORGATE_FUTURES

    legacy = {**base, "data_feed": "bloomberg"}
    assert spec_from_dict(legacy).data_feed is DataFeed.NORGATE_FUTURES


# ── Stage 5 back-compat lock: real on-disk specs + a spec built without data_feed ──

_SPECS_DIR = Path(__file__).resolve().parents[3] / "research" / "specs"


def _spec_json_paths() -> list[Path]:
    return sorted(_SPECS_DIR.glob("*.json"))


def test_specs_dir_has_the_three_back_compat_files():
    # Guards the parametrize below: if specs are added/removed this surfaces it explicitly
    # rather than silently shrinking coverage. The three checked-in specs all carry the
    # legacy ``data_feed="norgate_futures"`` token that must keep loading.
    names = {p.name for p in _spec_json_paths()}
    assert {"es_double7s_mr.json", "double7s_rerun.json", "bollinger_audnzd.json"} <= names


@pytest.mark.parametrize("spec_path", _spec_json_paths(), ids=lambda p: p.name)
def test_real_spec_files_load_with_optional_data_feed(spec_path: Path):
    # Every checked-in research/specs/*.json must still load+validate through the public
    # entrypoints now that data_feed is optional. (The 3 existing files carry
    # data_feed="norgate_futures"; an absent or legacy token must not break loading.)
    spec = load_spec(spec_path)
    assert spec.name  # validated, non-empty
    # data_feed is parsed tolerantly; whatever the file says, it resolves to a real DataFeed.
    assert isinstance(spec.data_feed, DataFeed)
    # load_spec (file) and spec_from_json (string) agree on the same file.
    assert spec_to_dict(load_spec(spec_path)) == spec_to_dict(
        spec_from_json(spec_path.read_text(encoding="utf-8-sig"))
    )
    # And the loaded spec round-trips losslessly back through dict + json.
    assert spec_to_dict(spec_from_dict(spec_to_dict(spec))) == spec_to_dict(spec)
    assert spec_to_dict(spec_from_json(spec_to_json(spec))) == spec_to_dict(spec)


def test_spec_built_without_data_feed_validates_and_round_trips():
    # Constructing a StrategySpec WITHOUT passing data_feed must validate (it defaults to
    # NORGATE_FUTURES) and round-trip through dict/json/file unchanged.
    spec = _minimal_spec()
    base = dict(
        name=spec.name,
        hypothesis=spec.hypothesis,
        author=spec.author,
        created=spec.created,
        tickers=spec.tickers,
        mode=spec.mode,
        timeframe=spec.timeframe,
        signal=spec.signal,
        direction=spec.direction,
        vault=spec.vault,
    )
    built = StrategySpec(**base)  # no data_feed kwarg at all
    assert built.data_feed is DataFeed.NORGATE_FUTURES

    _assert_specs_equal(spec_from_dict(spec_to_dict(built)), built)
    _assert_specs_equal(spec_from_json(spec_to_json(built)), built)
