import json

import numpy as np
import pandas as pd

from feature_selection.base_models import ContinuousBinningModel, RuleBasedModel
from feature_selection.base_models.utils import build_member_model_name


def test_new_model_names_exported() -> None:
    assert ContinuousBinningModel.__name__ == "ContinuousBinningModel"
    assert RuleBasedModel.__name__ == "RuleBasedModel"
    assert ContinuousBinningModel.model_type == "continuous_binning"
    assert RuleBasedModel.model_type == "rule_based"


def test_member_model_name_is_deterministic() -> None:
    assert (
        build_member_model_name("ewmac", "spanFast=32,spanSlow=128")
        == "ewmac::spanFast=32,spanSlow=128"
    )


def test_member_model_name_without_member_identity_keeps_single_name() -> None:
    assert build_member_model_name("ewmac", "") == "ewmac"


def test_binning_model_base_name_builder_uses_member_identity() -> None:
    model = ContinuousBinningModel(strategy="long")
    model.feature_column = "ewmac_signal_D_spanFast_32_spanSlow_128"

    assert (
        model.build_model_name("spanFast=32,spanSlow=128")
        == "ewmac_signal_D_spanFast_32_spanSlow_128_long::spanFast=32,spanSlow=128"
    )


def test_binning_model_base_name_builder_empty_identity_keeps_legacy_name() -> None:
    model = ContinuousBinningModel(strategy="long")
    model.feature_column = "ewmac_signal_D_spanFast_32_spanSlow_128"

    assert model.build_model_name("") == "ewmac_signal_D_spanFast_32_spanSlow_128_long"


def test_save_to_feature_list_appends_member_identity(tmp_path) -> None:
    feature = pd.Series(
        np.linspace(1.0, 2.0, 20),
        name="ewmac_signal_D_spanFast_32_spanSlow_128",
    )
    target = pd.Series(np.linspace(0.001, 0.01, 20), index=feature.index)

    model = ContinuousBinningModel(n_bins=3, strategy="long")
    model.fit(feature, target)

    control_file = tmp_path / "control.json"
    model.save_to_feature_list(
        filepath=str(control_file),
        member_identity="spanFast=32,spanSlow=128",
    )

    payload = json.loads(control_file.read_text())
    assert (
        payload["base_models"][0]["name"]
        == "ewmac_signal_D_spanFast_32_spanSlow_128_long::spanFast=32,spanSlow=128"
    )
