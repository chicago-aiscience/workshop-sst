"""Command-line interface for SST ML prediction workflow."""

import logging
from pathlib import Path

import pandas as pd
import typer

from .io import load_enso, load_sst
from .ml import predict_enso_from_sst
from .plot import make_ml_prediction_plot
from .transform import join_on_month, tidy

app = typer.Typer(
    help="SST CLI",
    no_args_is_help=True,
)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)


@app.command("predict")
def predict(
    sst: Path = Path("data/sst_sample.csv"),
    enso: Path = Path("data/nino34_sample.csv"),
    out_dir: Path = Path("artifacts"),
    start: str = "2000-01",
    n_lags: int = 3,
    test_size: float = 0.2,
    random_state: int = 1,
    model_path: Path | None = None,
) -> None:
    """Run machine learning prediction of ENSO from SST.

    Parameters
    ----------
    sst : pathlib.Path, default="data/sst_sample.csv"
        Location of the SST CSV file to ingest.
    enso : pathlib.Path, default="data/nino34_sample.csv"
        Location of the ENSO index CSV file to ingest.
    out_dir : pathlib.Path, default="artifacts"
        Directory where generated ML artifacts are written.
    start : str, default="2000-01"
        Earliest date to retain after joining the SST and ENSO data. Parsed
        to a timestamp via :func:`pandas.to_datetime`.
    n_lags : int, default=3
        Number of lag features to include (previous months' values).
    test_size : float, default=0.2
        Proportion of data to use for testing (between 0 and 1).
    random_state : int, default=42
        Random seed for reproducibility.
    model_path : pathlib.Path, optional
        Path to save the trained model. If not provided, saves to
        ``out_dir / "model.joblib"``.

    Returns
    -------
    None
        Writes ML prediction plot, metrics, and model to ``out_dir`` and prints
        their locations.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    if model_path is None:  # default model path
        model_path = out_dir / "model.joblib"

    sst_df = tidy(load_sst(sst), date_col="date", value_col="sst_c", roll=12)
    enso_df = tidy(load_enso(enso), date_col="date", value_col="nino34", roll=12)
    joined = join_on_month(sst_df, enso_df, start=start)

    logging.info("Training ML model to predict ENSO from SST...")
    results = predict_enso_from_sst(
        joined,
        n_lags=n_lags,
        test_size=test_size,
        random_state=random_state,
        model_path=model_path,
    )
    logging.info(
        f"Model performance: R² = {results['r2_score']: .3f}, RMSE = {results['rmse']: .3f}"
    )
    logging.info(f"Saved model to {model_path}")

    # Save predictions CSV
    predictions_path = out_dir / "ml_predictions.csv"
    predictions_value = results["predictions"]
    assert isinstance(predictions_value, pd.DataFrame), "predictions must be a DataFrame"
    predictions_df: pd.DataFrame = predictions_value
    predictions_df.to_csv(predictions_path, index=False)
    logging.info(f"Wrote {predictions_path}")

    # Save feature importance CSV
    importance_path = out_dir / "ml_feature_importance.csv"
    importance_value = results["feature_importance"]
    assert isinstance(importance_value, pd.DataFrame), "feature_importance must be a DataFrame"
    importance_df: pd.DataFrame = importance_value
    importance_df.to_csv(importance_path, index=False)
    logging.info(f"Wrote {importance_path}")

    # Save ML prediction plot
    fig = make_ml_prediction_plot(results)
    plot_path = out_dir / "ml_predictions.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logging.info(f"Wrote {plot_path}")


if __name__ == "__main__":
    app()
