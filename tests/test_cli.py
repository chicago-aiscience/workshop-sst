"""NumPy-style tests for the SST CLI interface."""

import pathlib
import subprocess
import sys

import pandas as pd


def test_cli_help() -> None:
    """Test that the CLI help command works correctly.

    Notes
    -----
    Verifies that the CLI can be invoked and shows help information.
    """
    cmd = [sys.executable, "-m", "sst.cli", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0
    assert "SST CLI" in result.stdout or "predict" in result.stdout


def test_cli_predict_command(tmp_path: pathlib.Path) -> None:
    """Test that the predict command runs and produces expected artifacts.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory used as the artifact output
        location for the CLI invocation.

    Notes
    -----
    Executes the predict command end-to-end with bundled sample data, asserting
    all ML outputs are created and the process exits successfully.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    out = tmp_path / "artifacts"
    out.mkdir()

    cmd = [
        sys.executable,
        "-m",
        "sst.cli",
        "--sst",
        str(data_dir / "sst_sample.csv"),
        "--enso",
        str(data_dir / "nino34_sample.csv"),
        "--out-dir",
        str(out),
        "--start",
        "2000-01",
        "--n-lags",
        "3",
        "--test-size",
        "0.2",
        "--random-state",
        "68",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"

    # Verify all expected artifacts are created
    assert (out / "ml_predictions.csv").exists()
    assert (out / "ml_feature_importance.csv").exists()
    assert (out / "ml_predictions.png").exists()
    assert (out / "model.joblib").exists()


def test_cli_predict_artifacts_content(tmp_path: pathlib.Path) -> None:
    """Test that the predict command produces artifacts with correct content.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory used as the artifact output
        location for the CLI invocation.

    Notes
    -----
    Verifies that the generated CSV files have the expected structure and
    contain valid data.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    out = tmp_path / "artifacts"
    out.mkdir()

    cmd = [
        sys.executable,
        "-m",
        "sst.cli",
        "--sst",
        str(data_dir / "sst_sample.csv"),
        "--enso",
        str(data_dir / "nino34_sample.csv"),
        "--out-dir",
        str(out),
        "--start",
        "2000-01",
        "--random-state",
        "36",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0

    # Verify predictions CSV structure
    predictions_path = out / "ml_predictions.csv"
    assert predictions_path.exists()
    predictions_df = pd.read_csv(predictions_path)
    assert "date" in predictions_df.columns
    assert "actual" in predictions_df.columns
    assert "predicted" in predictions_df.columns
    assert "residual" in predictions_df.columns
    assert len(predictions_df) > 0

    # Verify feature importance CSV structure
    importance_path = out / "ml_feature_importance.csv"
    assert importance_path.exists()
    importance_df = pd.read_csv(importance_path)
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert len(importance_df) > 0

    # Verify feature importance values are valid
    assert (importance_df["importance"] >= 0.0).all()
    assert (importance_df["importance"] <= 1.0).all()
    # Feature importance should sum to approximately 1.0
    assert abs(importance_df["importance"].sum() - 1.0) < 0.01


def test_cli_predict_with_custom_model_path(tmp_path: pathlib.Path) -> None:
    """Test that the predict command works with a custom model path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory used as the artifact output
        location for the CLI invocation.

    Notes
    -----
    Verifies that a custom model path can be specified and the model is
    saved to that location.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    out = tmp_path / "artifacts"
    out.mkdir()
    custom_model_path = tmp_path / "custom_model.joblib"

    cmd = [
        sys.executable,
        "-m",
        "sst.cli",
        "--sst",
        str(data_dir / "sst_sample.csv"),
        "--enso",
        str(data_dir / "nino34_sample.csv"),
        "--out-dir",
        str(out),
        "--start",
        "2000-01",
        "--model-path",
        str(custom_model_path),
        "--random-state",
        "78",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0
    assert custom_model_path.exists()
    # Default model path should not exist when custom path is provided
    assert not (out / "model.joblib").exists()
