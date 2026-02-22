def test_import_in_sample_packages() -> None:
    import feature_research.in_sample.continuous_binning.config as cb_config  # noqa: F401
    import feature_research.in_sample.continuous_binning.data_loader as cb_loader  # noqa: F401
    import feature_research.in_sample.continuous_binning.pipeline as cb_pipeline  # noqa: F401

    import feature_research.in_sample.rule_based.config as rb_config  # noqa: F401
    import feature_research.in_sample.rule_based.data_loader as rb_loader  # noqa: F401
    import feature_research.in_sample.rule_based.pipeline as rb_pipeline  # noqa: F401


def test_import_walkforward_entrypoints() -> None:
    from feature_research.walkforward.continuous_binning import run_walkforward as cb_walkforward  # noqa: F401
    from feature_research.walkforward.rule_based import run_walkforward as rb_walkforward  # noqa: F401
