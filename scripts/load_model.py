"""Load and explore a saved ML model artifact.

This script demonstrates how to:
1. Load a saved Random Forest model from a joblib file
2. Explore model attributes and properties
3. Make predictions with the loaded model
4. Inspect feature importance
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score

from sst.io import load_enso, load_sst
from sst.ml import _prep_data
from sst.transform import join_on_month, tidy


def load_and_explore_model(
    model_path: Path = Path("artifacts/model.joblib"),
    sst_path: Path = Path("data/sst_sample.csv"),
    enso_path: Path = Path("data/nino34_sample.csv"),
    start: str = "2000-01",
    n_lags: int = 3,
) -> None:
    """Load a saved model and explore its properties.

    Parameters
    ----------
    model_path : pathlib.Path, default="artifacts/model.joblib"
        Path to the saved model file.
    sst_path : pathlib.Path, default="data/sst_sample.csv"
        Path to SST data file (for preparing test data).
    enso_path : pathlib.Path, default="data/nino34_sample.csv"
        Path to ENSO data file (for preparing test data).
    start : str, default="2000-01"
        Start date for filtering data.
    n_lags : int, default=3
        Number of lag features (must match the model's training configuration).
    """
    print("=" * 70)
    print("LOADING SAVED MODEL")
    print("=" * 70)
    print(f"Model path: {model_path}")
    print()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    model = load(model_path)
    print("✓ Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
    print()

    # Explore model attributes
    print("=" * 70)
    print("MODEL ATTRIBUTES")
    print("=" * 70)
    print(f"  Number of trees: {model.n_estimators}")
    print(f"  Max depth: {model.max_depth}")
    print(f"  Random state: {model.random_state}")
    print(f"  Number of features: {model.n_features_in_}")
    print(f"  Feature names: {getattr(model, 'feature_names_in_', 'Not available')}")
    print()

    # Load and prepare data for prediction
    print("=" * 70)
    print("PREPARING DATA FOR PREDICTION")
    print("=" * 70)
    print("Loading and preparing data...")

    sst_df = tidy(load_sst(sst_path), date_col="date", value_col="sst_c", roll=12)
    enso_df = tidy(load_enso(enso_path), date_col="date", value_col="nino34", roll=12)
    joined = join_on_month(sst_df, enso_df, start=start)

    # Find actual column names
    target_col = [col for col in joined.columns if "nino34_roll" in col][0]
    feature_col = [col for col in joined.columns if "sst_c_roll" in col][0]

    # Prepare features (same as training)
    X, y, data, feature_names = _prep_data(
        joined, target_col=target_col, feature_col=feature_col, n_lags=n_lags
    )

    print(f"  Prepared {len(X)} samples with {X.shape[1]} features")
    print(f"  Feature names: {feature_names}")
    print()

    # Make predictions
    print("=" * 70)
    print("MAKING PREDICTIONS")
    print("=" * 70)
    predictions = model.predict(X)
    print(f"  Generated {len(predictions)} predictions")
    print()

    # Calculate metrics
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    print(f"  R² Score: {r2: .4f}")
    print(f"  RMSE: {rmse: .4f}")
    print()

    # Feature importance
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(importance_df.to_string(index=False))
    print()

    # Show some example predictions
    print("=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    example_df = pd.DataFrame(
        {
            "date": data.index[:10],
            "actual": y[:10],
            "predicted": predictions[:10],
            "error": y[:10] - predictions[:10],
        }
    )
    print(example_df.to_string(index=False))
    print()

    # Model summary
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"  Model file: {model_path}")
    print("  Model type: RandomForestRegressor")
    print(f"  Trees: {model.n_estimators}")
    print(f"  Max depth: {model.max_depth}")
    print(f"  Features: {model.n_features_in_}")
    print(f"  Performance: R² = {r2: .4f}, RMSE = {rmse: .4f}")
    print(
        f"  Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']: .4f})"
    )
    print()


if __name__ == "__main__":
    import sys

    # Allow command-line arguments
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/model.joblib")
    load_and_explore_model(model_path=model_path)
