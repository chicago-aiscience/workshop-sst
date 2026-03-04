"""Train SST-ENSO prediction model with Weights & Biases tracking.

This script demonstrates how to use the SST package with W&B for experiment tracking
on the free cloud tier at wandb.ai. It works on both laptops and HPC clusters with SLURM.

Usage:
    # Prerequisites: Free W&B account at https://wandb.ai
    # Login (one-time, interactive)
    wandb login

    # Or use API key (for CI, containers, shared machines)
    export WANDB_API_KEY=your_api_key_here

    # Basic usage
    python scripts/train_sst_wandb.py

    # With run name
    python scripts/train_sst_wandb.py --run-name "experiment_1"

    # With environment variables
    export WANDB_PROJECT="my_project"
    export DATA_ROOT="./data"
    python scripts/train_sst_wandb.py --run-name "test_run_1"

    # Offline mode (no network)
    WANDB_MODE=offline python scripts/train_sst_wandb.py
    # Later: wandb sync wandb/offline-run-* to upload

    # View results: Open the run URL printed at the end, or go to https://wandb.ai

Environment variables:
    WANDB_API_KEY         - API key for authentication (alternative to wandb login)
    WANDB_PROJECT         - Project name (default: sst_enso)
    WANDB_ENTITY          - Team/username (optional for personal accounts)
    WANDB_RUN_NAME        - Run name (overridden by --run-name)
    WANDB_MODE            - online (default), offline, or disabled
    WANDB_DATA_DIR        - Directory for W&B cache (default: ~/Library/Application Support/wandb)
    WANDB_MODEL_REGISTRY  - Registry path to link model (e.g. wandb-registry-workshop-sst/models)
    DATA_ROOT             - Data directory (default: ./data)
"""

import argparse
import json
import logging
import os
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

import wandb

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
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
            return pyproject_data.get("project", {}).get("version", "unknown")
    return "unknown"


def get_git_commit() -> dict[str, str]:
    """Get git commit information.

    Returns:
        Dictionary with git commit SHA, branch, and repo URL
    """
    git_info = {
        "commit": "unknown",
        "branch": "unknown",
        "repo_url": "unknown"
    }

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        git_info["commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        git_info["branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        git_info["repo_url"] = result.stdout.strip()

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return git_info


def setup_workspace(cfg: Config, version: str = "unknown") -> tuple[Path, Path]:
    """Setup workspace directory and save configuration.

    Args:
        cfg: Training configuration
        version: Package version to include in config (for artifact metadata)

    Returns:
        Tuple of (work_dir, config_path)
    """
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    config_path = work_dir / "config.json"
    config_dict = {**asdict(cfg), "package_version": version}
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    return work_dir, config_path


def load_and_prepare_data(cfg: Config, run) -> object:
    """Load and prepare SST and ENSO data.

    Args:
        cfg: Training configuration
        run: W&B run object for logging

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
    run.config.update({"total_samples": len(joined), "data_root": str(data_root)})

    return joined


def train_model(cfg: Config, data: object, work_dir: Path) -> tuple[dict, Path]:
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


def log_dvc_tags(run, data_dvc_path: Path, model_path: Path) -> None:
    """Run dvc add to update .dvc pointer files, then log content hashes to W&B.

    Args:
        run: W&B run object
        data_dvc_path: Path to the .dvc pointer file for the input data
        model_path: Path to the saved model file
    """
    data_file_path = data_dvc_path.with_suffix("")
    if data_file_path.exists():
        try:
            subprocess.run(["dvc", "add", str(data_file_path)], check=True, capture_output=True, timeout=30)
            with open(data_dvc_path) as f:
                dvc_info = yaml.safe_load(f)
            data_md5 = dvc_info.get("outs", [{}])[0].get("md5", "unknown")
            run.config.update({"dvc_data_md5": data_md5})
            logging.info(f"DVC data md5: {data_md5}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logging.warning(f"Could not run dvc add on data: {e}")

    if model_path.exists():
        model_dvc_path = Path(str(model_path) + ".dvc")
        try:
            subprocess.run(["dvc", "add", str(model_path)], check=True, capture_output=True, timeout=60)
            with open(model_dvc_path) as f:
                dvc_info = yaml.safe_load(f)
            model_md5 = dvc_info.get("outs", [{}])[0].get("md5", "unknown")
            run.config.update({"dvc_model_md5": model_md5})
            logging.info(f"DVC model md5: {model_md5}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logging.warning(f"Could not run dvc add on model: {e}")


def log_artifacts(run, results: dict, work_dir: Path, version: str) -> None:
    """Save and log artifacts to W&B.

    Args:
        run: W&B run object
        results: Training results dictionary
        work_dir: Working directory path
        version: Package version string for artifact metadata and aliases
    """
    version_alias = f"v{version}"

    # Save and log predictions
    predictions_path = work_dir / "ml_predictions.csv"
    results["predictions"].to_csv(predictions_path, index=False)
    artifact = wandb.Artifact(name="predictions", type="predictions")
    artifact.add_file(str(predictions_path))
    artifact.metadata["package_version"] = version
    run.log_artifact(artifact, aliases=[version_alias])

    # Save and log feature importance
    importance_path = work_dir / "ml_feature_importance.csv"
    results["feature_importance"].to_csv(importance_path, index=False)
    artifact = wandb.Artifact(name="feature_importance", type="features")
    artifact.add_file(str(importance_path))
    artifact.metadata["package_version"] = version
    run.log_artifact(artifact, aliases=[version_alias])

    # Save and log plot
    fig = make_ml_prediction_plot(results)
    plot_path = work_dir / "ml_predictions.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    artifact = wandb.Artifact(name="predictions_plot", type="plot")
    artifact.add_file(str(plot_path))
    artifact.metadata["package_version"] = version
    run.log_artifact(artifact, aliases=[version_alias])


def log_model_artifact(run, model_path: Path, version: str) -> None:
    """Log the trained model as a W&B artifact and optionally link to registry.

    Args:
        run: W&B run object
        model_path: Path to the saved model file
        version: Package version string
    """
    version_alias = f"v{version}"
    artifact = wandb.Artifact(name="sst_enso_predictor", type="model")
    artifact.add_file(str(model_path))
    artifact.metadata["package_version"] = version
    run.log_artifact(artifact, aliases=[version_alias])
    logging.info(f"✓ Model logged to W&B as 'sst_enso_predictor' (package version: {version})")

    # Link to Model Registry if configured
    registry_path = os.environ.get("WANDB_MODEL_REGISTRY")
    if registry_path:
        run.link_artifact(artifact, target_path=registry_path, aliases=[version_alias])
        logging.info(f"✓ Model linked to registry: {registry_path} (alias: {version_alias})")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train SST-ENSO prediction model with Weights & Biases tracking"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the W&B run (optional)"
    )
    return parser.parse_args()


def main() -> None:
    """Run the SST-ENSO prediction workflow with W&B tracking."""
    args = parse_args()
    cfg = Config()

    # Get package version and git info (needed for config and tags)
    version = get_package_version()
    git_info = get_git_commit()

    # Setup workspace and configuration
    work_dir, config_path = setup_workspace(cfg, version)

    # Authenticate via API key if set (for CI, containers, shared machines)
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    # W&B config and tags
    config = {
        "package_version": version,
        "seed": cfg.seed,
        "test_size": cfg.test_size,
        "n_lags": cfg.n_lags,
        "target_col": cfg.target_col,
        "feature_col": cfg.feature_col,
        "roll": cfg.roll,
    }
    tags = [
        f"v{version}",
        f"git:{git_info['commit'][:8]}",
        git_info["branch"],
    ]

    with wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sst_enso"),
        entity=os.environ.get("WANDB_ENTITY", None),
        name=args.run_name or os.environ.get("WANDB_RUN_NAME", None),
        config=config,
        tags=tags,
        notes=f"package_version={version}, git_commit={git_info['commit']}",
    ) as run:
        logging.info(f"Package version: {version}")
        logging.info(f"Git commit: {git_info['commit'][:8]}...")
        if args.run_name:
            logging.info(f"Run name: {args.run_name}")

        # Log config artifact
        artifact = wandb.Artifact(name="config", type="config")
        artifact.add_file(str(config_path))
        artifact.metadata["package_version"] = version
        run.log_artifact(artifact, aliases=[f"v{version}"])

        # Load and prepare data
        data = load_and_prepare_data(cfg, run)

        # Train model
        results, model_path = train_model(cfg, data, work_dir)
        logging.info(f"Model saved here: {model_path}")

        # Log metrics (summary used for cross-run charts on project dashboard)
        logging.info(f"R² score: {results['r2_score']:.4f}")
        logging.info(f"RMSE: {results['rmse']:.4f}")
        metrics = {"test_r2": results["r2_score"], "test_rmse": results["rmse"]}
        wandb.log(metrics)
        run.summary.update(metrics)

        # Log DVC content hashes
        data_root = Path(os.environ.get("DATA_ROOT", "./data"))
        log_dvc_tags(run, data_root / "sst_sample.csv.dvc", model_path)

        # Log artifacts
        log_artifacts(run, results, work_dir, version)

        # Log model
        log_model_artifact(run, model_path, version)

        # Summary
        logging.info(f"✓ Artifacts saved to: {work_dir}")
        if run.url:
            logging.info(f"✓ View results at: {run.url}")

    logging.info("Run complete.")


if __name__ == "__main__":
    main()
