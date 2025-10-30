"""Transform utilities for preparing SST and ENSO time series."""

import numpy as np
import pandas as pd


def tidy(df: pd.DataFrame, date_col: str, value_col: str, roll: int = 12) -> pd.DataFrame:
    """Create a tidy, chronologically ordered DataFrame with rolling means.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input data containing at least the date and value columns.
    date_col : str
        Name of the column with dates parsable by :func:`pandas.to_datetime`.
    value_col : str
        Name of the column with the measurement to smooth.
    roll : int, default=12
        Rolling window size (number of observations) used to compute the mean.

    Returns
    -------
    pandas.DataFrame
        Sorted copy of the original data with a new column containing the
        rolling mean named ``"{value_col}_roll{roll}"``.

    Examples
    --------
    >>> import pandas as pd
    >>> raw = pd.DataFrame({"date": ["2000-01-01", "2000-02-01"], "sst_c": [20.0, 20.1]})
    >>> tidy(raw, "date", "sst_c").columns.tolist()
    ['date', 'sst_c', 'sst_c_roll12']
    """

    out = df[[date_col, value_col]].copy()

    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).dropna()

    out[f"{value_col}_roll{roll}"] = out[value_col].rolling(roll, min_periods=1).mean()
    return out


def join_on_month(sst: pd.DataFrame, enso: pd.DataFrame, start: str | None = None) -> pd.DataFrame:
    """Join SST and ENSO records on their monthly ``date`` column.

    Parameters
    ----------
    sst : pandas.DataFrame
        Sea surface temperature observations produced by :func:`tidy`.
    enso : pandas.DataFrame
        ENSO index observations produced by :func:`tidy`.
    start : str, optional
        Earliest date to retain after joining (inclusive). Parsed with
        :func:`pandas.to_datetime` if provided.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the merged records, filtered to ``start`` when
        supplied, and indexed consecutively.

    Examples
    --------
    >>> import pandas as pd
    >>> sst = tidy(pd.DataFrame({"date": ["2000-01-01"], "sst_c": [20.0]}), "date", "sst_c")
    >>> enso = tidy(pd.DataFrame({"date": ["2000-01-01"], "nino34": [0.5]}), "date", "nino34")
    >>> join_on_month(sst, enso).columns.tolist()
    ['date', 'sst_c', 'sst_c_roll12', 'nino34', 'nino34_roll12']
    """

    df = pd.merge(sst, enso, on="date", how="left")
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    return df.reset_index(drop=True)


def _simple_trend(y: pd.Series, per: str = "decade") -> float:
    """Compute a linear trend in the provided series.

    Parameters
    ----------
    y : pandas.Series
        Series of measurements indexed by :class:`pandas.DatetimeIndex`.
    per : {"decade", "year"}, default="decade"
        Unit for the returned slope. Values other than ``"decade"`` fall back
        to per-year units.

    Returns
    -------
    float
        Estimated slope, scaled to the requested unit.

    Raises
    ------
    ValueError
        If the series index is not a :class:`~pandas.DatetimeIndex`.

    Examples
    --------
    >>> import pandas as pd
    >>> series = pd.Series([1.0, 1.1], index=pd.to_datetime(["2000-01-01", "2001-01-01"]))
    >>> round(_simple_trend(series), 6)
    0.1
    """

    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("Series index must be DatetimeIndex.")

    t_years = (y.index - y.index.min()).days / 365.25
    slope = float(np.polyfit(t_years, y.values, 1)[0])  # units per year

    return slope * 10.0 if per == "decade" else slope


def _delta_last_year(s: pd.Series) -> float:
    """Average difference between the most recent and prior 12 months.

    Parameters
    ----------
    s : pandas.Series
        Rolling metric series indexed by date.

    Returns
    -------
    float
        Mean of the newest year minus the mean of the preceding year.

    Examples
    --------
    >>> import pandas as pd
    >>> series = pd.Series(
    ...     [1.0] * 12 + [1.2] * 12,
    ...     index=pd.date_range("2000-01-01", periods=24, freq="MS"),
    ... )
    >>> round(_delta_last_year(series), 1)
    0.2
    """

    s = s.dropna()
    if len(s) < 24:
        return float("nan")
    return float(s.tail(12).mean() - s.tail(24).head(12).mean())


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize rolling SST and ENSO time series with key indicators.

    Parameters
    ----------
    df : pandas.DataFrame
        Joined SST and ENSO tidy data that contains a ``date`` column along
        with at least one rolling SST column (``sst_c_roll*``) and one rolling
        ENSO column (``nino34_roll*``).

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame containing trend, delta, correlation, and record
        count statistics for the supplied series.

    Examples
    --------
    >>> import pandas as pd
    >>> joined = join_on_month(
    ...     tidy(pd.DataFrame({"date": ["2000-01-01"], "sst_c": [20.0]}), "date", "sst_c"),
    ...     tidy(pd.DataFrame({"date": ["2000-01-01"], "nino34": [0.5]}), "date", "nino34"),
    ... )
    >>> metrics(joined).columns.tolist()
    ['sst_trend_c_per_decade', 'delta_sst_last_yr_c', 'delta_enso_last_yr', 'corr_sst_enso_roll', 'n_months']
    """

    d = df.set_index("date")

    sst_col = (
        "sst_c_roll12"
        if "sst_c_roll12" in d.columns
        else [c for c in d.columns if c.startswith("sst_c_roll")][0]
    )
    enso_col = (
        "nino34_roll12"
        if "nino34_roll12" in d.columns
        else [c for c in d.columns if c.startswith("nino34_roll")][0]
    )

    sst_trend_c_per_dec = _simple_trend(d[sst_col].dropna(), per="decade")

    delta_sst_lastyr = _delta_last_year(d[sst_col])
    delta_enso_lastyr = _delta_last_year(d[enso_col])

    corr = d[sst_col].corr(d[enso_col])

    return pd.DataFrame(
        [
            {
                "sst_trend_c_per_decade": (
                    round(sst_trend_c_per_dec, 3) if pd.notna(sst_trend_c_per_dec) else None
                ),
                "delta_sst_last_yr_c": (
                    round(delta_sst_lastyr, 3) if pd.notna(delta_sst_lastyr) else None
                ),
                "delta_enso_last_yr": (
                    round(delta_enso_lastyr, 3) if pd.notna(delta_enso_lastyr) else None
                ),
                "corr_sst_enso_roll": round(float(corr), 3) if pd.notna(corr) else None,
                "n_months": int(len(d)),
            }
        ]
    )
