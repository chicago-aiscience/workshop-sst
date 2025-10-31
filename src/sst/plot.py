"""Plotting utilities for SST and ENSO trend visualizations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_trend_plot(df: pd.DataFrame) -> plt.Figure:
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

    ax1.plot(df["date"], df["sst_c_roll12"], label="SST (°C, roll12)", color="blue")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("SST (°C)")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["nino34_roll12"], label="Niño 3.4 (roll12)", color="orange")
    ax2.set_ylabel("Niño 3.4 index")

    ax1.set_title("SST and ENSO (12‑mo rolling means)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    fig.tight_layout()

    return fig


def make_corr_plot(joined: pd.DataFrame) -> plt.Figure:
    """Visualize rolling SST and ENSO relationship from a joined dataset.

    Parameters
    ----------
    joined : pandas.DataFrame
        Output of :func:`sst.transform.join_on_month` containing date, SST
        rolling means, and ENSO rolling means. The frame must include
        ``sst_c_roll12`` and ``nino34_roll12`` columns.

    Returns
    -------
    matplotlib.figure.Figure
        Scatter plot figure showing the correlation between rolling SST and
        ENSO values.

    Raises
    ------
    ValueError
        If ``joined`` is empty or lacks the required rolling columns.
    """

    if joined.empty:
        raise ValueError("Joined DataFrame must contain at least one row.")

    required_cols = {"sst_c_roll12", "nino34_roll12"}
    if not required_cols.issubset(joined.columns):
        missing = required_cols.difference(joined.columns)
        raise ValueError(f"Joined DataFrame is missing required columns: {sorted(missing)}")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(
        data=joined,
        x="nino34_roll12",
        y="sst_c_roll12",
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "black"},
        ax=ax,
    )

    ax.set_title("Correlation of 12-Month Rolling SST vs ENSO")
    ax.set_xlabel("Niño 3.4 (roll12)")
    ax.set_ylabel("SST (°C, roll12)")

    fig.tight_layout()
    return fig
