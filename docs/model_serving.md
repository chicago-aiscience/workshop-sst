# Model Serving with Docker

Serve the SST-ENSO model via a REST API using a Docker image published to GitHub Container Registry (GHCR).

## Quick Start

```bash
# Pull the image
docker pull ghcr.io/chicago-aiscience/workshop-sst-serve:latest

# Run (default model, port 8000)
docker run -p 8000:8000 ghcr.io/chicago-aiscience/workshop-sst-serve:latest

# Health check
curl http://localhost:8000/health

# Model info (feature names, schema)
curl http://localhost:8000/model-info

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[0.5, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2]]}'
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|--------------|
| `MODEL_PATH` | `/app/model/model.joblib` | Path to the model file inside the container |
| `PORT` | `8000` | Server port |

### Volume Mount: Custom Model

To use your own trained model instead of the default:

```bash
docker run -p 8000:8000 \
  -v /path/to/your/model.joblib:/app/model/model.joblib:ro \
  ghcr.io/chicago-aiscience/workshop-sst-serve:latest
```

### Override Port

```bash
docker run -p 9000:9000 -e PORT=9000 ghcr.io/chicago-aiscience/workshop-sst-serve:latest
```

---

## API Endpoints

### GET /health

Liveness check. Returns `{"status": "ok"}`.

### GET /model-info

Returns feature names and schema for the loaded model:

```json
{
  "model_path": "/app/model/model.joblib",
  "n_features": 7,
  "feature_names": ["sst_c_roll_12", "sst_c_roll_12_lag_1", ...]
}
```

### POST /predict

Predict ENSO index from SST features.

**Request formats:**

1. **Instances** – A list of input samples. Each element is one sample: an array of feature values in the order the model expects. For the default model (`n_lags=3`), there are 7 features: `sst_c_roll_12`, `sst_c_roll_12_lag_1`, `sst_c_roll_12_lag_2`, `sst_c_roll_12_lag_3`, `nino34_roll_12_lag_1`, `nino34_roll_12_lag_2`, `nino34_roll_12_lag_3`. Use `GET /model-info` to confirm the exact feature order for your model.

```json
{"instances": [[0.5, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2]]}
```

Multiple samples: `{"instances": [[...], [...], ...]}` returns one prediction per sample.

**Example with multiple samples:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [
    [0.5, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2],
    [0.6, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3],
    [0.4, 0.2, 0.1, 0.0, 0.3, 0.2, 0.1]
  ]}'
```

Response: `{"predictions": [0.42, 0.51, 0.38]}` (one value per input sample).

**How to obtain feature data**

The feature values come from SST (sea surface temperature) and ENSO (Niño 3.4) data. Use the `sst` package to load CSVs, apply a 12-month rolling mean, and create lag features:

```python
from sst.io import load_sst, load_enso
from sst.ml import _prep_data
from sst.transform import join_on_month, tidy

# Load and prepare (same pipeline as training)
sst_df = tidy(load_sst("data/sst_sample.csv"), date_col="date", value_col="sst_c", roll=12)
enso_df = tidy(load_enso("data/nino34_sample.csv"), date_col="date", value_col="nino34", roll=12)
joined = join_on_month(sst_df, enso_df, start="2000-01")

# Extract feature arrays for prediction
X, y, data, feature_names = _prep_data(
    joined, target_col="nino34_roll_12", feature_col="sst_c_roll_12", n_lags=3
)

# X is a 2D array; each row is one sample. Convert to list for the API:
instances = X.tolist()
# Then: requests.post(url, json={"instances": instances})
```

Sample data is in `data/sst_sample.csv` and `data/nino34_sample.csv`. For custom data, use CSVs with `date`, `sst_c` (or equivalent), and `nino34` columns.

2. **Dataframe split** (MLflow-compatible):

```json
{
  "dataframe_split": {
    "columns": ["sst_c_roll_12", "sst_c_roll_12_lag_1", ...],
    "data": [[0.5, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2]]
  }
}
```

**Response:**

```json
{"predictions": [0.42]}
```

---

## Building Locally

```bash
# Build the serve image
docker build -f Dockerfile.serve -t workshop-sst-serve:local .

# Run
docker run -p 8000:8000 workshop-sst-serve:local
```

---

## Python Client Example

```python
import requests

url = "http://localhost:8000/predict"

# Get feature names first
info = requests.get("http://localhost:8000/model-info").json()
print("Features:", info["feature_names"])

# Predict
response = requests.post(
    url,
    json={"instances": [[0.5, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2]]},
)
predictions = response.json()["predictions"]
print("Predictions:", predictions)
```

---

## Image Location

The serve image is published to GHCR by the deploy workflow:

- **Image:** `ghcr.io/<org>/workshop-sst-serve` (e.g. `ghcr.io/chicago-aiscience/workshop-sst-serve`)
- **Tags:** `latest`, version tags (e.g. `0.6.2`)
- **Trigger:** Pushes to `dev` or `main` branches
