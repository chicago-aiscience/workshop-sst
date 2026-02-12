"""Train SST-ENSO prediction model with MLflow tracking.

This script demonstrates how to use the SST package with MLflow for experiment tracking.
It works on both laptops and HPC clusters with SLURM.

Usage:
    # Basic usage (laptop)
    python scripts/mlflow/train_sst_mlflow.py

    # With run name argument
    python scripts/mlflow/train_sst_mlflow.py --run-name "experiment_1"

    # With environment variables
    export MLFLOW_EXPERIMENT_NAME="my_experiment"
    export MLFLOW_TRACKING_DIR="./runs"
    export DATA_ROOT="./data"
    python scripts/mlflow/train_sst_mlflow.py --run-name "test_run_1"

    # View results
    mlflow ui --backend-store-uri runs/sst_enso/mlruns
"""

import argparse
import json
import logging
import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sst.io import load_enso, load_sst
from sst.ml import predict_enso_from_sst
from sst.plot import make_ml_prediction_plot
from sst.transform import join_on_month, tidy


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    level=logging.INFO
)



@dataclass
class Config:
    """Training configuration for SST-ENSO prediction."""

    seed: int = 42
    test_size: float = 0.2
    n_lags: int = 3
    work_dir: str = "runs/sst_enso"
    target_col: str = "nino34_roll_12"
    feature_col: str = "sst_c_roll_12"
    roll: int = 12


def get_package_version() -> str:
    """Get version from pyproject.toml.

    Returns:
        Package version string
    """
    # Script is at scripts/train_sst_mlflow.py, so need 2 parent calls to reach repo root
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
            return pyproject_data.get("project", {}).get("version", "unknown")
    return "unknown"


def setup_workspace(cfg: Config) -> tuple[Path, Path]:
    """Setup workspace directory and save configuration.

    Args:
        cfg: Training configuration

    Returns:
        Tuple of (work_dir, config_path)
    """
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    config_path = work_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return work_dir, config_path


def setup_mlflow(cfg: Config, work_dir: Path, config_path: Path, run_name: str | None = None) -> tuple[str, str]:
    """Setup MLflow tracking, experiment, and log initial parameters.

    Args:
        cfg: Training configuration
        work_dir: Working directory path
        config_path: Configuration file path
        run_name: Optional name for the MLflow run

    Returns:
        Tuple of (MLflow tracking directory path, package version)
    """
    mlruns_dir = os.environ.get("MLFLOW_TRACKING_DIR", str((work_dir / "mlruns").absolute()))
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "sst_enso"))

    # Use provided run_name, fall back to env var, then None
    final_run_name = run_name or os.environ.get("MLFLOW_RUN_NAME", None)
    mlflow.start_run(run_name=final_run_name)

    # Get package version
    version = get_package_version()

    # Log parameters
    mlflow.log_params(
        {
            "seed": cfg.seed,
            "test_size": cfg.test_size,
            "n_lags": cfg.n_lags,
            "target_col": cfg.target_col,
            "feature_col": cfg.feature_col,
            "roll": cfg.roll,
        }
    )

    # Tag run with version
    mlflow.set_tag("package_version", version)
    mlflow.set_tag("mlflow.source.name", f"sst-v{version}")

    mlflow.log_artifact(str(config_path), artifact_path="config")

    return mlruns_dir, version


def load_and_prepare_data(cfg: Config):
    """Load and prepare SST and ENSO data.

    Args:
        cfg: Training configuration

    Returns:
        Joined dataframe with SST and ENSO data
    """
    logging.info("Loading data...")
    data_root = Path(os.environ.get("DATA_ROOT", "./data"))
    sst_path = data_root / "sst_sample.csv"
    enso_path = data_root / "nino34_sample.csv"

    if not sst_path.exists():
        raise FileNotFoundError(f"SST data not found at {sst_path}")
    if not enso_path.exists():
        raise FileNotFoundError(f"ENSO data not found at {enso_path}")

    sst_df = tidy(load_sst(sst_path), date_col="date", value_col="sst_c", roll=cfg.roll)
    enso_df = tidy(load_enso(enso_path), date_col="date", value_col="nino34", roll=cfg.roll)
    joined = join_on_month(sst_df, enso_df, start="2000-01")

    logging.info(f"Total samples: {len(joined)}")
    mlflow.set_tag("total_samples", len(joined))
    mlflow.set_tag("data_root", str(data_root))

    # Log dataset to MLflow
    dataset = mlflow.data.from_pandas(
        joined,
        source=str(data_root),
        name="sst_enso_dataset",
        targets=cfg.target_col
    )
    mlflow.log_input(dataset, context="training")

    return joined


def train_model(cfg: Config, data, work_dir: Path) -> tuple[dict, Path]:
    """Train the SST-ENSO prediction model.

    Args:
        cfg: Training configuration
        data: Prepared training data
        work_dir: Working directory path

    Returns:
        Tuple of (results dict, model_path)
    """
    logging.info("Training model...")
    model_path = work_dir / "model.joblib"
    results = predict_enso_from_sst(
        data,
        target_col=cfg.target_col,
        feature_col=cfg.feature_col,
        n_lags=cfg.n_lags,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        model_path=model_path,
    )
    return results, model_path


def log_metrics(results: dict) -> None:
    """Log evaluation metrics to MLflow.

    Args:
        results: Training results dictionary
    """
    logging.info(f"R² score: {results['r2_score']:.4f}")
    logging.info(f"RMSE: {results['rmse']:.4f}")

    mlflow.log_metric("test_r2", results["r2_score"])
    mlflow.log_metric("test_rmse", results["rmse"])


def log_artifacts(results: dict, work_dir: Path) -> None:
    """Save and log artifacts to MLflow.

    Args:
        results: Training results dictionary
        work_dir: Working directory path
    """
    # Save and log predictions
    predictions_path = work_dir / "ml_predictions.csv"
    results["predictions"].to_csv(predictions_path, index=False)
    mlflow.log_artifact(str(predictions_path), artifact_path="predictions")

    # Save and log feature importance
    importance_path = work_dir / "ml_feature_importance.csv"
    results["feature_importance"].to_csv(importance_path, index=False)
    mlflow.log_artifact(str(importance_path), artifact_path="features")

    # Save and log plot
    fig = make_ml_prediction_plot(results)
    plot_path = work_dir / "ml_predictions.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path), artifact_path="plots")


def register_model(cfg: Config, results: dict, model_path: Path, version: str) -> None:
    """Register the trained model to MLflow Model Registry.

    Args:
        cfg: Training configuration
        results: Training results dictionary
        model_path: Path to the saved model file
        version: Package version string
    """
    # Load the trained model
    model = joblib.load(model_path)

    # Create input example and signature for model schema
    feature_cols = [f"{cfg.feature_col}_lag_{i}" for i in range(cfg.n_lags)]
    input_example = results["predictions"][feature_cols].head(5) if all(
        col in results["predictions"].columns for col in feature_cols
    ) else None

    # Infer model signature (input/output schema)
    signature = infer_signature(
        results["predictions"][feature_cols] if input_example is not None else results["predictions"].iloc[:, :cfg.n_lags],
        results["predictions"][["predicted"]]
    )

    # Log the model with MLflow's sklearn flavor and register it
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="sst_enso_predictor",
    )

    # Tag the registered model version with package version
    client = MlflowClient()

    # Get the model version that was just registered
    model_version = model_info.registered_model_version
    if model_version:
        client.set_model_version_tag(
            name="sst_enso_predictor",
            version=model_version,
            key="package_version",
            value=version
        )

    logging.info(f"\n✓ Model registered to MLflow Model Registry as 'sst_enso_predictor' (package version: {version})")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train SST-ENSO prediction model with MLflow tracking"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the MLflow run (optional)"
    )
    return parser.parse_args()


def main() -> None:
    """Run the SST-ENSO prediction workflow with MLflow tracking."""
    args = parse_args()
    cfg = Config()

    # Setup workspace and configuration
    work_dir, config_path = setup_workspace(cfg)

    # Setup MLflow tracking
    mlruns_dir, version = setup_mlflow(cfg, work_dir, config_path, run_name=args.run_name)
    logging.info(f"Package version: {version}")
    if args.run_name:
        logging.info(f"Run name: {args.run_name}")

    # Load and prepare data
    data = load_and_prepare_data(cfg)

    # Train model
    results, model_path = train_model(cfg, data, work_dir)

    # Log metrics
    log_metrics(results)

    # Log artifacts
    log_artifacts(results, work_dir)

    # Register model
    register_model(cfg, results, model_path, version)

    # logging.info summary
    logging.info(f"\n✓ Artifacts saved to: {work_dir}")
    logging.info(f"✓ MLflow tracking URI: {mlruns_dir}")
    logging.info("\nTo view results, run:")
    logging.info(f"  mlflow ui --backend-store-uri {mlruns_dir}")

    mlflow.end_run()


if __name__ == "__main__":
    main()
