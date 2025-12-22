"""Tests for machine learning functionality."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from sst.io import load_enso, load_sst
from sst.ml import _prep_data, predict_enso_from_sst
from sst.transform import join_on_month, tidy


def test_data_loading_and_joining() -> None:
    """Test that input data can be loaded and joined correctly.

    Notes
    -----
    Validates that SST and ENSO data can be loaded from CSV files,
    processed with tidy, and joined correctly for ML model input.
    """
    # Load data
    sst_path = Path("data/sst_sample.csv")
    enso_path = Path("data/nino34_sample.csv")

    sst_df = load_sst(sst_path)
    enso_df = load_enso(enso_path)

    # Verify data structure
    assert "date" in sst_df.columns
    assert "sst_c" in sst_df.columns
    assert "date" in enso_df.columns
    assert "nino34" in enso_df.columns

    # Process with tidy
    sst_tidy = tidy(sst_df, "date", "sst_c", roll=12)
    enso_tidy = tidy(enso_df, "date", "nino34", roll=12)

    # Verify tidy output (column names use format: {value_col}_roll_{roll})
    assert any("sst_c_roll" in col for col in sst_tidy.columns)
    assert any("nino34_roll" in col for col in enso_tidy.columns)
    assert len(sst_tidy) > 0
    assert len(enso_tidy) > 0

    # Join data
    joined = join_on_month(sst_tidy, enso_tidy, start="2000-01")

    # Verify joined data structure
    assert "date" in joined.columns
    # Find the actual column names
    sst_roll_col = [col for col in joined.columns if "sst_c_roll" in col][0]
    enso_roll_col = [col for col in joined.columns if "nino34_roll" in col][0]
    assert sst_roll_col in joined.columns
    assert enso_roll_col in joined.columns
    assert len(joined) > 0
    assert not joined["date"].isna().any()
    assert not joined[sst_roll_col].isna().any()
    assert not joined[enso_roll_col].isna().any()


def test_data_formatting_for_model() -> None:
    """Test that the data is formatted correctly for the model.

    Notes
    -----
    Validates that _prep_data creates the correct feature structure
    with lag features and proper data types for ML model input.
    """
    # Create sample data using tidy to get correct column names
    dates = pd.date_range("2000-01", periods=20, freq="ME")
    sst_raw = pd.DataFrame({"date": dates, "sst_c": np.random.randn(20) + 20.0})
    enso_raw = pd.DataFrame({"date": dates, "nino34": np.random.randn(20)})

    sst_tidy = tidy(sst_raw, "date", "sst_c", roll=12)
    enso_tidy = tidy(enso_raw, "date", "nino34", roll=12)
    joined = join_on_month(sst_tidy, enso_tidy)

    # Find actual column names (format: {value_col}_roll_{roll})
    target_col = [col for col in joined.columns if "nino34_roll" in col][0]
    feature_col = [col for col in joined.columns if "sst_c_roll" in col][0]

    # Prepare data for model
    X, y, data, feature_names = _prep_data(
        joined, target_col=target_col, feature_col=feature_col, n_lags=3
    )

    # Verify feature array structure
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert X.shape[0] > 0  # Should have rows after dropping NaN
    assert X.shape[1] == 1 + (3 * 2)  # 1 base feature + 3 SST lags + 3 ENSO lags

    # Verify target array structure
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert len(y) == X.shape[0]

    # Verify feature names
    assert len(feature_names) == 7  # 1 base + 3 SST lags + 3 ENSO lags
    assert feature_col in feature_names
    assert f"{feature_col}_lag_1" in feature_names
    assert f"{feature_col}_lag_2" in feature_names
    assert f"{feature_col}_lag_3" in feature_names
    assert f"{target_col}_lag_1" in feature_names
    assert f"{target_col}_lag_2" in feature_names
    assert f"{target_col}_lag_3" in feature_names

    # Verify data DataFrame structure
    assert isinstance(data, pd.DataFrame)
    assert len(data) == X.shape[0]
    assert all(col in data.columns for col in feature_names)

    # Verify no NaN values in features or target
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_model_and_results() -> None:
    """Test the model and results by comparing expected results.

    Notes
    -----
    Validates that the ML model produces results with expected structure
    and reasonable performance metrics.
    """
    # Load and prepare data
    sst_path = Path("data/sst_sample.csv")
    enso_path = Path("data/nino34_sample.csv")

    sst_df = tidy(load_sst(sst_path), "date", "sst_c", roll=12)
    enso_df = tidy(load_enso(enso_path), "date", "nino34", roll=12)
    joined = join_on_month(sst_df, enso_df, start="2000-01")

    # Run prediction with fixed random state for reproducibility
    results = predict_enso_from_sst(joined, random_state=42, n_lags=3, test_size=0.2)

    # Verify results dictionary structure
    assert isinstance(results, dict)
    assert "r2_score" in results
    assert "rmse" in results
    assert "predictions" in results
    assert "feature_importance" in results

    # Verify R² score
    r2 = results["r2_score"]
    assert isinstance(r2, float)
    assert -1.0 <= r2 <= 1.0  # R² can be negative for very poor models

    # Verify RMSE
    rmse = results["rmse"]
    assert isinstance(rmse, float)
    assert rmse >= 0.0

    # Verify predictions DataFrame
    predictions = results["predictions"]
    assert isinstance(predictions, pd.DataFrame)
    assert "date" in predictions.columns
    assert "actual" in predictions.columns
    assert "predicted" in predictions.columns
    assert "residual" in predictions.columns
    assert len(predictions) > 0

    # Verify predictions structure
    assert predictions["date"].dtype == "datetime64[ns]"
    assert predictions["actual"].dtype in [np.float64, np.float32]
    assert predictions["predicted"].dtype in [np.float64, np.float32]
    assert predictions["residual"].dtype in [np.float64, np.float32]

    # Verify residuals are calculated correctly
    expected_residuals = predictions["actual"] - predictions["predicted"]
    residual_values = np.asarray(predictions["residual"].values)
    expected_values = np.asarray(expected_residuals.values)
    np.testing.assert_array_almost_equal(residual_values, expected_values, decimal=10)

    # Verify feature importance DataFrame
    importance = results["feature_importance"]
    assert isinstance(importance, pd.DataFrame)
    assert "feature" in importance.columns
    assert "importance" in importance.columns
    assert len(importance) > 0

    # Verify feature importance values
    assert (importance["importance"] >= 0.0).all()
    assert (importance["importance"] <= 1.0).all()
    # Feature importance should sum to approximately 1.0
    assert abs(importance["importance"].sum() - 1.0) < 0.01

    # Verify model can be saved and loaded
    model_path = Path("artifacts/test_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    results_with_model = predict_enso_from_sst(
        joined, random_state=42, n_lags=3, test_size=0.2, model_path=model_path
    )

    assert "model" in results_with_model
    assert model_path.exists()

    # Load and verify model
    loaded_model = load(model_path)
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")

    # Clean up
    if model_path.exists():
        model_path.unlink()
