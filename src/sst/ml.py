"""Machine learning utilities for predicting SST-ENSO relationships."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _prep_data(
    df: pd.DataFrame, target_col: str, feature_col: str, n_lags: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Prepare data for machine learning model.

    Parameters
    ----------
    df : pandas.DataFrame
        Joined SST and ENSO tidy data that contains a ``date`` column along
        with at least one rolling SST column and one rolling ENSO column.
    target_col : str
        Name of the ENSO column to predict.
    feature_col : str
        Name of the SST column to use as the primary feature.
    n_lags : int
        Number of lag features to include (previous months' values).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]
        Tuple of features and target arrays, the prepared data, and the feature names.

    Examples
    --------
    >>> import pandas as pd
    >>> from sst.transform import join_on_month, tidy
    >>> from sst.io import load_sst, load_enso
    >>> sst_df = tidy(load_sst("data/sst_sample.csv"), "date", "sst_c")
    >>> enso_df = tidy(load_enso("data/nino34_sample.csv"), "date", "nino34")
    >>> joined = join_on_month(sst_df, enso_df)
    >>> X, y = _prep_data(joined, "nino34_roll_12", "sst_c_roll_12", 3)
    """

    data = df.set_index("date").sort_index()
    data = data[[feature_col, target_col]].dropna()

    if len(data) < n_lags + 10:
        raise ValueError(
            f"Insufficient data: need at least {n_lags + 10} observations, got {len(data)}"
        )

    # Create lag features
    feature_names = [feature_col]

    for lag in range(1, n_lags + 1):
        lag_col = f"{feature_col}_lag_{lag}"
        data[lag_col] = data[feature_col].shift(lag)
        feature_names.append(lag_col)

    # Also include target lag features (autoregressive)
    for lag in range(1, n_lags + 1):
        lag_col = f"{target_col}_lag_{lag}"
        data[lag_col] = data[target_col].shift(lag)
        feature_names.append(lag_col)

    # Drop rows with NaN (from lagging)
    data = data.dropna()

    # Prepare features and target
    X = data[feature_names].values
    y = np.asarray(data[target_col].values)
    return X, y, data, feature_names


def _collect_results(
    model: RandomForestRegressor,
    y_pred_test: np.ndarray,
    y_test: np.ndarray,
    data: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, float | pd.DataFrame | RandomForestRegressor]:
    """Collect results from the machine learning model.

    Returns
    -------
    dict[str, float | pd.DataFrame | RandomForestRegressor]
        Dictionary containing the results.
    """
    # Calculate metrics
    r2 = float(r2_score(y_test, y_pred_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    # Create predictions DataFrame
    test_dates = data.index[-len(y_test) :]
    predictions_df = pd.DataFrame(
        {
            "date": test_dates,
            "actual": y_test,
            "predicted": y_pred_test,
            "residual": y_test - y_pred_test,
        }
    )

    # Feature importance
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    result_dict = {
        "r2_score": r2,
        "rmse": rmse,
        "predictions": predictions_df,
        "feature_importance": importance_df,
    }
    return result_dict


def predict_enso_from_sst(
    df: pd.DataFrame,
    target_col: str = "nino34_roll_12",
    feature_col: str = "sst_c_roll_12",
    test_size: float = 0.2,
    n_lags: int = 3,
    random_state: int = 42,
    model_path: Path | None = None,
) -> dict[str, float | pd.DataFrame | RandomForestRegressor]:
    """Predict ENSO index from SST using a Random Forest model with lag features.

    This function creates a machine learning model to predict ENSO (Niño 3.4 index)
    from sea surface temperature data. It includes lag features to capture temporal
    dependencies in the time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Joined SST and ENSO tidy data that contains a ``date`` column along
        with at least one rolling SST column and one rolling ENSO column.
    target_col : str, default="nino34_roll_12"
        Name of the ENSO column to predict.
    feature_col : str, default="sst_c_roll_12"
        Name of the SST column to use as the primary feature.
    test_size : float, default=0.2
        Proportion of data to use for testing (between 0 and 1).
    n_lags : int, default=3
        Number of lag features to include (previous months' values).
    random_state : int, default=42
        Random seed for reproducibility.
    model_path : pathlib.Path, optional
        If provided, save the trained model to this path using joblib.

    Returns
    -------
    dict[str, float | pandas.DataFrame | RandomForestRegressor]
        Dictionary containing:
        - ``r2_score``: R² score on test set
        - ``rmse``: Root mean squared error on test set
        - ``predictions``: DataFrame with date, actual, and predicted values
        - ``feature_importance``: DataFrame with feature importance scores
        - ``model``: Trained RandomForestRegressor model (if model_path provided)

    Examples
    --------
    >>> import pandas as pd
    >>> from sst.transform import join_on_month, tidy
    >>> from sst.io import load_sst, load_enso
    >>> sst_df = tidy(load_sst("data/sst_sample.csv"), "date", "sst_c")
    >>> enso_df = tidy(load_enso("data/nino34_sample.csv"), "date", "nino34")
    >>> joined = join_on_month(sst_df, enso_df)
    >>> results = predict_enso_from_sst(joined, model_path=Path("model.joblib"))
    >>> "r2_score" in results
    True
    """
    # Prepare the data for training
    X, y, data, feature_names = _prep_data(df, target_col, feature_col, n_lags)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10)
    model.fit(X_train, y_train)

    # Save model if path provided
    if model_path is not None:
        dump(model, model_path)
        result_dict = {"model": model}
    else:
        result_dict = {}

    # Make predictions
    _ = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    result_dict.update(_collect_results(model, y_pred_test, y_test, data, feature_names))
    return result_dict
