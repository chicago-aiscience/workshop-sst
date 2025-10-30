"""IO helpers for loading SST and ENSO data sets."""

from pathlib import Path

import pandas as pd


def load_sst(path: Path) -> pd.DataFrame:
    """Load sea surface temperature observations from disk.

    Parameters
    ----------
    path : pathlib.Path
        File system location of a CSV file containing ``date`` and ``sst_c``
        columns.

    Returns
    -------
    pandas.DataFrame
        Parsed SST table with original column names and dtypes.
    """
    return pd.read_csv(path)


def load_enso(path: Path) -> pd.DataFrame:
    """Load ENSO index observations from disk.

    Parameters
    ----------
    path : pathlib.Path
        File system location of a CSV file containing ``date`` and ``nino34``
        columns.

    Returns
    -------
    pandas.DataFrame
        Parsed ENSO index table with original column names and dtypes.
    """
    return pd.read_csv(path)
