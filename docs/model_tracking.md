# Model Tracking: MLflow, W&B, and DVC

This document describes how MLflow, Weights & Biases (W&B), and DVC are used in this repository for experiment tracking, model versioning, and data traceability.

## Overview

| Tool | Purpose | Script |
|------|---------|--------|
| **MLflow** | Local experiment tracking, model registry, artifacts | `scripts/train_sst_mlflow.py` |
| **W&B** | Cloud experiment tracking (free tier at wandb.ai) | `scripts/train_sst_wandb.py` |
| **DVC** | Data and model versioning; content hashes logged to MLflow/W&B | Both scripts |

Both training scripts share the same workflow: load SST/ENSO data, train a Random Forest model, log metrics and artifacts. The main difference is where results are stored—locally (MLflow) or in the cloud (W&B).

---

## MLflow

MLflow provides local experiment tracking with a file-based backend. No account or network is required.

### Quick Start

```bash
# Train a model
python scripts/train_sst_mlflow.py

# With custom run name
python scripts/train_sst_mlflow.py --run-name "baseline_model"

# View results
mlflow ui --backend-store-uri runs/sst_enso/mlruns
# Open http://localhost:5000
```

### What Gets Tracked

- **Parameters**: `seed`, `test_size`, `n_lags`, `target_col`, `feature_col`, `roll`
- **Metrics**: R² score, RMSE
- **Tags**: `package_version`, `git_commit`, `git_branch`, `git_repo_url`, `total_samples`, `data_root`, `dvc_data_md5`, `dvc_model_md5`
- **Artifacts**: Config JSON, predictions CSV, feature importance CSV, plot PNG
- **Model**: Registered to MLflow Model Registry as `sst_enso_predictor`

### Configuration

```bash
export MLFLOW_EXPERIMENT_NAME="my_experiment"
export MLFLOW_TRACKING_DIR="./runs"
export MLFLOW_RUN_NAME="exp_1"
export DATA_ROOT="./data"
python scripts/train_sst_mlflow.py --run-name "exp_1"
```

### Model Registry

Models are registered as `sst_enso_predictor`. Load a model:

```python
import mlflow.sklearn

# Latest version
model = mlflow.sklearn.load_model("models:/sst_enso_predictor/latest")

# Specific version
model = mlflow.sklearn.load_model("models:/sst_enso_predictor/1")
```

### Workflow Structure

```
runs/sst_enso/
├── mlruns/                    # MLflow tracking data
│   └── 0/                     # Experiment ID
│       ├── <run_id>/          # Individual runs
│       └── models/            # Model Registry
├── model.joblib               # Trained model (local copy)
├── ml_predictions.csv
├── ml_feature_importance.csv
└── ml_predictions.png
```

---

## Weights & Biases (W&B)

W&B provides cloud-based experiment tracking on the free tier at [wandb.ai](https://wandb.ai).

### Quick Start

```bash
# Prerequisites: Free account at https://wandb.ai

# Login (one-time)
wandb login

# Or use API key (for CI, containers, shared machines)
export WANDB_API_KEY=your_api_key_here

# Train
python scripts/train_sst_wandb.py

# With run name
python scripts/train_sst_wandb.py --run-name "experiment_1"
```

### What Gets Tracked

- **Config**: Same parameters as MLflow (`seed`, `test_size`, `n_lags`, etc.)
- **Metrics**: R² score, RMSE (via `wandb.log`)
- **Tags**: Package version, git commit, branch
- **Artifacts**: Config, predictions, feature importance, plot, model (as W&B Artifacts)
- **Config metadata**: `total_samples`, `data_root`, `dvc_data_md5`, `dvc_model_md5`

### Model Registry

To link the trained model to a W&B Model Registry, set `WANDB_MODEL_REGISTRY` to the registry path (e.g. `wandb-registry-workshop-sst/models`). The path format is `wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}`. Create the registry and collection in the W&B UI first. The linked model is assigned an alias matching the package version (e.g. `v0.6.2`) so you can reference it as `wandb-registry-workshop-sst/models:v0.6.2`.

```bash
export WANDB_MODEL_REGISTRY="wandb-registry-workshop-sst/models"
python scripts/train_sst_wandb.py
```

### Artifact Versioning

Each artifact includes `package_version` in its metadata and a custom alias `v{version}` (e.g., `v0.6.2`). You can:

- **View metadata**: Open an artifact in the W&B UI and check the metadata panel for `package_version`.
- **Reference by app version**: Use `artifact_name:v0.6.2` to fetch the artifact produced by that package version, e.g. `sst_enso_predictor:v0.6.2`.

### Configuration

```bash
export WANDB_PROJECT="my_project"
export WANDB_ENTITY="my_team"        # Optional for personal accounts
export WANDB_RUN_NAME="exp_1"
export WANDB_MODE="offline"         # online (default), offline, or disabled
export WANDB_DATA_DIR="./wandb_cache"  # For restricted write environments
export WANDB_MODEL_REGISTRY="wandb-registry-workshop-sst/models"  # Link model to registry
export DATA_ROOT="./data"
python scripts/train_sst_wandb.py --run-name "exp_1"
```

### Offline Mode

```bash
WANDB_MODE=offline python scripts/train_sst_wandb.py
# Later, upload: wandb sync wandb/offline-run-*
```

### View Results

Open the run URL printed at the end of training, or go to [wandb.ai](https://wandb.ai) and select your project.

### Chart Metrics Across Runs

To compare R² and RMSE across experiments:

1. Go to your **project** page (e.g. `sst_enso`), not a single run.
2. Click **Add panel** (or **+** in the workspace).
3. Choose a chart type such as **Parallel coordinates**, **Scatter**, or **Bar chart**.
4. Set the **X-axis** or **Y-axis** to `test_r2` or `test_rmse`.
5. Optionally group or color by config (e.g. `n_lags`, `test_size`).

The runs table already lists `test_r2` and `test_rmse` per run; panels let you visualize them across runs.

---

## DVC

DVC (Data Version Control) is used for versioning large files—datasets and trained models—without storing them in Git. Small `.dvc` pointer files are committed instead.

### How It's Used in This Repo

1. **Data files**: `data/sst_sample.csv` and `data/nino34_sample.csv` are tracked via `.dvc` files (`sst_sample.csv.dvc`, `nino34_sample.csv.dvc`).

2. **Content hashes in experiment tracking**: Both training scripts run `dvc add` on the input data and trained model, then log the MD5 content hashes to MLflow (as tags) or W&B (as config). This links each experiment run to the exact data and model versions used.

### DVC Workflow

```bash
# Add or update data tracking
dvc add data/sst_sample.csv
dvc add data/nino34_sample.csv

# Pull data (if using remote storage)
dvc pull

# Push data (if using remote storage)
dvc push
```

### .dvc File Structure

Each `.dvc` file stores metadata about the tracked file:

```yaml
outs:
- md5: d08ae445bfa70901879bfe45ae78de40
  size: 2160
  hash: md5
  path: sst_sample.csv
```

The `md5` hash is what gets logged to MLflow (`dvc_data_md5`, `dvc_model_md5`) or W&B config for traceability.

### Integration with MLflow and W&B

When you run `train_sst_mlflow.py` or `train_sst_wandb.py`:

1. The script runs `dvc add` on the data files (if they exist) and the saved model.
2. It reads the MD5 hashes from the `.dvc` files.
3. It logs these hashes to the experiment run so you can trace which data and model produced each result.

If DVC is not installed or `dvc add` fails (e.g., in a restricted environment), the scripts log a warning and continue without the DVC hashes.

---

## Loading and Using Models

### From MLflow Registry

```python
import mlflow.sklearn

# Set tracking URI (or use MLFLOW_TRACKING_URI env var)
mlflow.set_tracking_uri("file:runs/sst_enso/mlruns")

# Load latest version
model = mlflow.sklearn.load_model("models:/sst_enso_predictor/latest")

# Load specific version
model = mlflow.sklearn.load_model("models:/sst_enso_predictor/1")

# Make predictions
predictions = model.predict(X)
```

### From W&B Registry

```python
import joblib
import wandb
from pathlib import Path

# Option 1: Within a run (latest version)
with wandb.init(project="sst_enso") as run:
    artifact = run.use_artifact("wandb-registry-workshop-sst/models:latest")
    artifact_dir = artifact.download()
    model = joblib.load(Path(artifact_dir) / "model.joblib")

# Option 2: Specific alias (e.g. package version v0.6.2)
with wandb.init(project="sst_enso") as run:
    artifact = run.use_artifact("wandb-registry-workshop-sst/models:v0.6.2")
    artifact_dir = artifact.download()
    model = joblib.load(Path(artifact_dir) / "model.joblib")

# Option 3: Without an active run (Public API)
api = wandb.Api()
# Full path: entity/project/artifact_name:alias (see artifact's Full name in W&B UI)
artifact = api.artifact("your-entity/wandb-registry-workshop-sst/models:latest")
artifact_dir = artifact.download()
model = joblib.load(Path(artifact_dir) / "model.joblib")
```

**Note:** The artifact path format is `wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}` when used within a run. For the Public API, use the full path shown in the W&B UI (e.g. `entity/wandb-registry-{name}/{collection}:alias`). Use `latest` for the most recent version, `v0`/`v1` for version indices, or a custom alias like `v0.6.2`.

---

## Sharing Models: Best Practices

| Aspect | MLflow | W&B |
|--------|--------|-----|
| **Location** | Local (file-based) or remote (DB, S3) | Cloud (wandb.ai) |
| **Sharing** | Share tracking URI; users need access to backend store | Share project URL; team members need W&B account or API key |
| **Teams** | Use remote model registry on shared storage (S3, DB) | Use W&B Teams with entity/project; roles: Admin, Member, View-Only |
| **Access control** | Backend storage permissions (e.g. S3 IAM) | W&B project visibility (private, team, public) |
| **CI/CD** | Set `MLFLOW_TRACKING_URI` and `MLFLOW_REGISTRY_URI` | Set `WANDB_API_KEY`; reference artifact by registry path |
| **Offline** | Fully local; no network needed | `WANDB_MODE=offline`; sync later with `wandb sync` |

**Recommendations:**

- **MLflow**: Best for local or on-premise workflows, air-gapped environments, and when you control the backend store. Use a remote tracking server or S3-backed registry for team sharing.
- **W&B**: Best for cloud collaboration, distributed teams, and rich dashboards. Use the Model Registry for staging/production promotion; set `WANDB_API_KEY` in CI/CD.
- **Both**: Log model metadata (package version, git commit) for traceability. Use DVC hashes for data/model lineage when available.

---

## Comparing Experiments

### MLflow

```bash
python scripts/train_sst_mlflow.py --run-name "3_lags"
# Edit Config: n_lags = 5
python scripts/train_sst_mlflow.py --run-name "5_lags"
# Edit Config: test_size = 0.3
python scripts/train_sst_mlflow.py --run-name "split_30"
```

Use the MLflow UI **Compare** button to view metrics side-by-side.

### W&B

Run the same experiments with `train_sst_wandb.py`. Use the W&B dashboard to filter, group, and compare runs by config and metrics.

---

## Config Parameters

Both scripts use the same `Config` dataclass. Edit it in the script to change hyperparameters:

```python
@dataclass
class Config:
    seed: int = 42
    test_size: float = 0.2
    n_lags: int = 3      # Try 5, 7, 10
    roll: int = 12
```

---

## Tips

- **Run names**: Use descriptive names like `"7_lags_split_0.2"` instead of auto-generated IDs.
- **Version tags**: Each run is tagged with package version and git commit for traceability.
- **DVC hashes**: When available, `dvc_data_md5` and `dvc_model_md5` let you reproduce the exact data and model for any run.
- **MLflow vs W&B**: Use MLflow for fully local, offline workflows; use W&B for cloud collaboration and dashboards.

---

## References

- [MLflow documentation](https://mlflow.org/docs/latest/index.html)
- [Weights & Biases documentation](https://docs.wandb.ai/)
- [DVC documentation](https://dvc.org/doc)
- [Model Serving with Docker](model_serving.md) – Serve the model via REST API from a container
