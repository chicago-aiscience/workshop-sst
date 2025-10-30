"""NumPy-style tests for SST transforms."""

import pandas as pd

from sst.transform import join_on_month, metrics, tidy

def test_tidy_adds_roll12():
    """Check that `tidy` adds a rolling column while preserving row count.

    Notes
    -----
    Uses two records to confirm the rolling mean column exists and the output
    retains exactly the original number of entries.
    """

    df = pd.DataFrame({"date": ["2000-01-01", "2000-02-01"], "sst_c": [20.0, 20.1]})
    out = tidy(df, "date", "sst_c")

    assert "sst_c_roll12" in out.columns
    assert len(out) == 2


def test_join_and_metrics_shape():
    """Ensure joined data produces metric summary with expected fields.

    Notes
    -----
    Validates the joined DataFrame includes both rolling series and that
    `metrics` returns the key statistic columns required downstream.
    """

    sst = pd.DataFrame({"date": ["2000-01-01", "2000-02-01"], "sst_c": [20.0, 20.1]})
    enso = pd.DataFrame({"date": ["2000-01-01", "2000-02-01"], "nino34": [0.5, 0.2]})

    joined = join_on_month(tidy(sst, "date", "sst_c"), tidy(enso, "date", "nino34"))
    summary = metrics(joined)

    assert {"sst_trend_c_per_decade", "corr_sst_enso_roll"}.issubset(summary.columns)
