from __future__ import annotations

import pandas as pd

from research.portfolio.holdout.rolling_eval import _stitch_frames, _stitch_series


def test_stitch_series_deduplicates_overlapping_index() -> None:
    first = pd.Series([0.01, 0.02], index=pd.to_datetime(["2023-01-03", "2023-01-04"]))
    second = pd.Series([0.03, 0.04], index=pd.to_datetime(["2023-01-04", "2023-01-05"]))
    stitched = _stitch_series([first, second])
    assert stitched.shape[0] == 3
    assert float(stitched.loc[pd.Timestamp("2023-01-04")]) == 0.03


def test_stitch_frames_aligns_columns() -> None:
    first = pd.DataFrame({"a": [0.1, 0.2]}, index=pd.to_datetime(["2023-01-03", "2023-01-04"]))
    second = pd.DataFrame({"a": [0.3]}, index=pd.to_datetime(["2023-01-05"]))
    stitched = _stitch_frames([first, second])
    assert list(stitched.columns) == ["a"]
    assert stitched.shape[0] == 3
