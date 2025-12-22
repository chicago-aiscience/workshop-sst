"""Plotting utilities for SST and ENSO trend visualizations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _plot_ax_1(ax1: plt.Axes, predictions_df: pd.DataFrame, r2_score: float, rmse: float) -> None:
    """Plot the first panel of the ML prediction plot."""
    ax1.plot(
        predictions_df["date"],
        predictions_df["actual"],
        label="Actual",
        color="blue",
        alpha=0.7,
        linewidth=1.5,
    )
    ax1.plot(
        predictions_df["date"],
        predictions_df["predicted"],
        label="Predicted",
        color="orange",
        alpha=0.7,
        linewidth=1.5,
        linestyle="--",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Niño 3.4 Index")
    ax1.set_title(f"Predictions Over Time\nR² = {r2_score: .3f}, RMSE = {rmse: .3f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Rotate and resize date labels for better readability
    ax1.tick_params(axis="x", rotation=90, labelsize=9)


def _plot_ax_2(ax2: plt.Axes, predictions_df: pd.DataFrame) -> None:
    """Plot the second panel of the ML prediction plot."""
    ax2.scatter(
        predictions_df["actual"],
        predictions_df["predicted"],
        alpha=0.6,
        s=30,
        color="steelblue",
    )
    # Add perfect prediction line
    min_val = min(predictions_df["actual"].min(), predictions_df["predicted"].min())
    max_val = max(predictions_df["actual"].max(), predictions_df["predicted"].max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")
    ax2.set_xlabel("Actual Niño 3.4 Index")
    ax2.set_ylabel("Predicted Niño 3.4 Index")
    ax2.set_title("Actual vs Predicted")
    ax2.legend()
    ax2.grid(True, alpha=0.3)


def _plot_ax_3(ax3: plt.Axes, importance_df: pd.DataFrame) -> None:
    """Plot the third panel of the ML prediction plot."""
    top_features = importance_df.head(10)  # Show top 10 features

    # Check if there's a dominant feature (>90% importance)
    max_importance = top_features["importance"].max()
    if max_importance > 0.9:
        # If one feature dominates, use log scale or show percentages
        # Use percentage scale to make smaller values visible
        importance_pct = top_features["importance"] * 100
        bars = ax3.barh(  # noqa: F841
            range(len(top_features)), importance_pct, color="coral", alpha=0.7
        )

        # Add value labels to the right of y-axis with white box background
        for i, (idx, row) in enumerate(top_features.iterrows()):
            value = row["importance"] * 100
            label = f"{value: .2f}%" if value > 1 else f"{row['importance']: .4f}"
            ax3.text(
                105,  # Position to the right of the x-axis limit
                i,
                label,
                ha="left",
                va="center",
                fontsize=7,
                fontweight="bold" if i == 0 else "normal",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
            )

        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels(top_features["feature"], fontsize=8, rotation=45, ha="right")
        ax3.set_xlabel("Importance (%)")
        ax3.set_xlim(0, 105)  # Slightly beyond 100% for labels
    else:
        # Normal case: all features visible on 0-1 scale
        bars = ax3.barh(  # noqa: F841
            range(len(top_features)), top_features["importance"], color="coral"
        )

        # Add value labels to the right of y-axis with white box background
        max_val = top_features["importance"].max()
        for i, (idx, row) in enumerate(top_features.iterrows()):
            value = row["importance"]
            label = f"{value: .4f}"
            ax3.text(
                max_val * 1.05,  # Position slightly to the right of max bar
                i,
                label,
                ha="left",
                va="center",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
            )

        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels(top_features["feature"], fontsize=8, rotation=45, ha="right")
        ax3.set_xlabel("Importance")

    ax3.set_title("Top 10 Feature Importance")
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis="x")


def make_ml_prediction_plot(results: dict[str, float | pd.DataFrame]) -> plt.Figure:
    """Visualize machine learning prediction results for ENSO from SST.

    Creates a multi-panel figure showing:
    1. Time series of actual vs predicted ENSO values
    2. Scatter plot of actual vs predicted values
    3. Feature importance bar plot

    Parameters
    ----------
    results : dict[str, float | pandas.DataFrame]
        Dictionary returned by :func:`sst.ml.predict_enso_from_sst` containing:
        - ``predictions``: DataFrame with date, actual, predicted, residual columns
        - ``feature_importance``: DataFrame with feature and importance columns
        - ``r2_score``: R² score (float)
        - ``rmse``: Root mean squared error (float)

    Returns
    -------
    matplotlib.figure.Figure
        Figure with three subplots showing prediction results.

    Examples
    --------
    >>> from sst.ml import predict_enso_from_sst
    >>> from sst.transform import join_on_month, tidy
    >>> from sst.io import load_sst, load_enso
    >>> sst_df = tidy(load_sst("data/sst_sample.csv"), "date", "sst_c")
    >>> enso_df = tidy(load_enso("data/nino34_sample.csv"), "date", "nino34")
    >>> joined = join_on_month(sst_df, enso_df)
    >>> results = predict_enso_from_sst(joined)
    >>> fig = make_ml_prediction_plot(results)
    >>> fig.savefig("ml_predictions.png")
    """
    predictions_df = results["predictions"]
    importance_df = results["feature_importance"]
    r2_score_val = results["r2_score"]
    rmse_val = results["rmse"]

    # Type assertions for mypy
    assert isinstance(predictions_df, pd.DataFrame), "predictions must be a DataFrame"
    assert isinstance(importance_df, pd.DataFrame), "feature_importance must be a DataFrame"
    assert isinstance(r2_score_val, float), "r2_score must be a float"
    assert isinstance(rmse_val, float), "rmse must be a float"

    r2_score: float = r2_score_val
    rmse: float = rmse_val

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 5))

    # Panel 1: Time series of actual vs predicted
    ax1 = plt.subplot(1, 3, 1)
    _plot_ax_1(ax1, predictions_df, r2_score, rmse)

    # Panel 2: Scatter plot of actual vs predicted
    ax2 = plt.subplot(1, 3, 2)
    _plot_ax_2(ax2, predictions_df)

    # Panel 3: Feature importance
    ax3 = plt.subplot(1, 3, 3)
    _plot_ax_3(ax3, importance_df)

    plt.tight_layout()
    return fig
