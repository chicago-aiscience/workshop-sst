"""Plotting utilities for SST and ENSO trend visualizations."""

import matplotlib.pyplot as plt
import pandas as pd

def make_trend_plot(df: pd.DataFrame):
    """Create a dual-axis plot for rolling SST and ENSO series.

    Parameters
    ----------
    df : pandas.DataFrame
        Joined data containing ``date``, ``sst_c_roll12``, and
        ``nino34_roll12`` columns.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object with SST on the primary axis and ENSO on a secondary
        axis.
    """

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(df["date"], df["sst_c_roll12"], label="SST (°C, roll12)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("SST (°C)")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["nino34_roll12"], label="Niño 3.4 (roll12)")
    ax2.set_ylabel("Niño 3.4 index")

    ax1.set_title("SST and ENSO (12‑mo rolling means)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    fig.tight_layout()

    return fig
