from pathlib import Path
from datetime import datetime

from ensemble.weight_layer import WeightLayerConfig
from feature_research.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
    WeightLayerAlgorithm,
)
from feature_research.walkforward.stable_region_selection import StableRegionConfig


SELECTION_DOC = Path(
    "docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md"
)
WEIGHT_LAYER_DOC = Path("docs/library/Ensemble/weight_layer.md")


def _default_walkforward_config() -> WalkforwardResearchConfig:
    return WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2021, 1, 1),
    )


def test_selection_doc_states_selection_stage_does_not_average_forecasts() -> None:
    content = SELECTION_DOC.read_text(encoding="utf-8")
    assert "Selection is selection-only: no forecast averaging is performed at the selection stage." in content


def test_weight_layer_doc_states_weight_layer_is_primary_combiner() -> None:
    content = WEIGHT_LAYER_DOC.read_text(encoding="utf-8")
    assert "The WeightLayer is the primary forecast combiner." in content


def test_docs_describe_selection_and_weighting_config_knobs() -> None:
    selection_content = SELECTION_DOC.read_text(encoding="utf-8")
    weight_layer_content = WEIGHT_LAYER_DOC.read_text(encoding="utf-8")
    defaults = _default_walkforward_config()

    selection_default = (
        defaults.selection_method.value
        if isinstance(defaults.selection_method, WalkforwardSelectionMethod)
        else str(defaults.selection_method)
    )
    weighting_default = WeightLayerConfig().weighting_method

    assert f"| `selection_method` | `{selection_default}` |" in selection_content
    assert f"| `weighting_method` | `{weighting_default}` |" in weight_layer_content

    for method in WalkforwardSelectionMethod:
        assert f"`{method.value}`" in selection_content

    for method in WeightLayerAlgorithm:
        assert f"`{method.value}`" in weight_layer_content


def test_stable_region_doc_describes_trade_frequency_as_upstream_prefilter() -> None:
    selection_content = SELECTION_DOC.read_text(encoding="utf-8")
    config = _default_walkforward_config()
    stable_config = StableRegionConfig()

    assert config.trade_freq_min == 0.05
    assert not hasattr(stable_config, "trade_freq_min")
    assert "upstream walkforward prefilter" in selection_content.lower()
    assert "WalkforwardResearchConfig" in selection_content
