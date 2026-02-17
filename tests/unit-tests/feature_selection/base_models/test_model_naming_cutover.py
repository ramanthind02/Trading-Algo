from feature_selection.base_models import ContinuousBinningModel, RuleBasedModel


def test_new_model_names_exported() -> None:
    assert ContinuousBinningModel.__name__ == "ContinuousBinningModel"
    assert RuleBasedModel.__name__ == "RuleBasedModel"
    assert ContinuousBinningModel.model_type == "continuous_binning"
    assert RuleBasedModel.model_type == "rule_based"
