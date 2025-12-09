"""Transform utilities for preparing SST and ENSO time series."""

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
        rolling mean named ``"{value_col}_roll_{roll}"``.

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

    out[f"{value_col}_roll_{roll}"] = out[value_col].rolling(roll, min_periods=1).mean()
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
