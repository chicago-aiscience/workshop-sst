"""Serve the SST-ENSO model via a REST API.

Environment variables:
    MODEL_PATH: Path to model file (default: /app/model/model.joblib)
    PORT: Server port (default: 8000)

Endpoints:
    GET /health: Liveness check
    GET /model-info: Feature names and schema
    POST /predict: Predict ENSO from SST features
"""

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from joblib import load

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/model/model.joblib"))

app = FastAPI(
    title="SST-ENSO Model API",
    description="Predict ENSO (Niño 3.4) index from Sea Surface Temperature lag features",
    version="0.1.0",
)

_model = None


def get_model():
    """Load model on first request."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Mount a model or set MODEL_PATH.",
            )
        _model = load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Return feature names and schema for the loaded model."""
    try:
        model = get_model()
    except HTTPException:
        raise
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        feature_names = list(feature_names)
    return {
        "model_path": str(MODEL_PATH),
        "n_features": getattr(model, "n_features_in_", None),
        "feature_names": feature_names,
    }


@app.post("/predict")
async def predict(request: Request):
    """Predict ENSO index from SST features.

    Accepts either:
    - {"instances": [[f1, f2, ...], ...]} - features in model order
    - {"dataframe_split": {"columns": [...], "data": [[...], ...]}}
    """
    body = await request.json()
    try:
        model = get_model()
    except HTTPException:
        raise

    feature_names = getattr(model, "feature_names_in_", None)
    n_features = getattr(model, "n_features_in_", None)

    X = None

    if "instances" in body:
        instances = body["instances"]
        X = np.array(instances, dtype=np.float64)
    elif "dataframe_split" in body:
        split = body["dataframe_split"]
        columns = split.get("columns", [])
        data = split.get("data", [])
        if feature_names is not None and columns != list(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Columns must match model features: {list(feature_names)}",
            )
        X = np.array(data, dtype=np.float64)
    else:
        raise HTTPException(
            status_code=400,
            detail="Request must include 'instances' or 'dataframe_split'",
        )

    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="Input must be 2D array")

    if n_features is not None and X.shape[1] != n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {n_features} features, got {X.shape[1]}",
        )

    predictions = model.predict(X)
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
