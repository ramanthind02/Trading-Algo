import py_compile


def test_binning_analysis_entrypoint_compiles() -> None:
    py_compile.compile(
        "feature_research/in_sample/binning_analysis.py", doraise=True
    )
