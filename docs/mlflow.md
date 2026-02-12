# MLflow Experiment Tracking

Track, version, and deploy your SST-ENSO prediction models with MLflow.

## Quick Start

### 1. Train a Model

```bash
# Basic training
python scripts/mlflow/train_sst_mlflow.py

# With custom run name
python scripts/mlflow/train_sst_mlflow.py --run-name "baseline_model"
```

### 2. View Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri runs/sst_enso/mlruns

# Open http://localhost:5000
```

### 3. Export Best Model

```bash
# Export the best performing model
python scripts/mlflow/export_best_model.py
```

The best model will be exported to `models/best_model/` with all artifacts.

## What Gets Tracked?

### Automatically Logged

- **Parameters**: `n_lags`, `test_size`, `seed`, `roll`
- **Metrics**: R² score, RMSE
- **Package Version**: From `pyproject.toml` (e.g., `0.6.2`)
- **Model**: Registered to MLflow Model Registry
- **Artifacts**: Predictions, feature importance, plots

### Tags

Each run is tagged with:
- `package_version`: Code version used
- `total_samples`: Dataset size
- `data_root`: Data location

## Configuration

### Environment Variables

```bash
# Customize experiment tracking
export MLFLOW_EXPERIMENT_NAME="my_experiment"
export MLFLOW_TRACKING_DIR="./runs"
export DATA_ROOT="./data"

python scripts/mlflow/train_sst_mlflow.py --run-name "exp_1"
```

### Config Parameters

Edit the `Config` class in `train_sst_mlflow.py`:

```python
@dataclass
class Config:
    seed: int = 42
    test_size: float = 0.2
    n_lags: int = 3           # Try 5, 7, 10
    roll: int = 12
```

## Model Registry

Models are automatically registered to the MLflow Model Registry as `sst_enso_predictor`.

### View Registered Models

In the MLflow UI:
1. Click the **Models** tab
2. Select **sst_enso_predictor**
3. View all model versions and their tags

### Load a Registered Model

```python
import mlflow.sklearn

# Load latest version
model = mlflow.sklearn.load_model("models:/sst_enso_predictor/latest")

# Load specific version
model = mlflow.sklearn.load_model("models:/sst_enso_predictor/1")

# Make predictions
predictions = model.predict(data)
```

## Comparing Experiments

Run multiple experiments and compare in the UI:

```bash
# Experiment 1: Baseline
python scripts/mlflow/train_sst_mlflow.py --run-name "3_lags"

# Experiment 2: More lags (edit Config: n_lags = 5)
python scripts/mlflow/train_sst_mlflow.py --run-name "5_lags"

# Experiment 3: Different split (edit Config: test_size = 0.3)
python scripts/mlflow/train_sst_mlflow.py --run-name "split_30"
```

In MLflow UI, use the **Compare** button to view metrics side-by-side.

## Deployment

### Serve Model Locally

```bash
# Serve the latest model
mlflow models serve -m "models:/sst_enso_predictor/latest" -p 5000

# Make predictions
curl http://localhost:5000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {
    "columns": ["sst_c_roll_12_lag_0", "sst_c_roll_12_lag_1", "sst_c_roll_12_lag_2"],
    "data": [[0.5, 0.3, 0.2]]
  }}'
```

### Export for Deployment

```bash
# Export best model
python scripts/mlflow/export_best_model.py

# Output directory: models/best_model/
# - model.joblib
# - predictions.csv
# - feature_importance.csv
# - plot.png
# - metadata.json
```

## Workflow Structure

```
runs/sst_enso/
├── mlruns/                    # MLflow tracking data
│   └── 0/                     # Experiment ID
│       ├── <run_id>/          # Individual runs
│       └── models/            # Model Registry
├── model.joblib               # Trained model (local copy)
├── ml_predictions.csv         # Predictions
└── ml_predictions.png         # Visualization
```

## Tips

- **Run Names**: Use descriptive names like `"7_lags_split_0.2"` instead of auto-generated IDs
- **Version Tags**: Each run and model version is tagged with package version for traceability
- **Model Registry**: Use the registry to track which models are in staging/production
- **Artifacts**: All plots, CSVs, and models are automatically saved

## Next Steps

1. Run the training script and explore the MLflow UI
2. Try different parameter combinations
3. Compare results across runs
4. Export and deploy your best model

See the [full MLflow documentation](https://mlflow.org/docs/latest/index.html) for advanced features.
