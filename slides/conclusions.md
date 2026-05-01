---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.1rem;
  }
  h1 { color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 0.3em; }
  h2 { color: #2874a6; }
  code { background: #eaf2ff; border-radius: 4px; padding: 0.1em 0.3em; }
  table { font-size: 0.9rem; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2em; }
---

# Experiment Tracking, Model Versioning & Serving
## Workshop: SST-ENSO Prediction

**Tools covered:** DVC · MLflow · Weights & Biases · Docker

---

## Agenda

1. The challenge: reproducibility in ML
2. Data versioning with DVC
3. Experiment tracking: MLflow vs W&B
4. Model sharing & registries
5. Model serving with Docker
6. Conclusions & recommendations

---

## The Reproducibility Problem

When you train a model, you need to track:

- **Data** — which exact dataset was used?
- **Code** — which version of the training script?
- **Parameters** — what hyperparameters?
- **Results** — what metrics did we get?
- **Model artifact** — which `.joblib` file produced that result?

> Without tracking, you can't reproduce, compare, or share your work reliably.

---

## Data Versioning with DVC

DVC tracks large files (data, models) outside of Git by storing **content hashes** (MD5) in small `.dvc` pointer files committed to the repo.

```yaml
# sst_sample.csv.dvc
outs:
- md5: d08ae445bfa70901879bfe45ae78de40
  size: 2160
  path: sst_sample.csv
```

These hashes are then **logged directly to MLflow and W&B**, creating a traceable link:

```
Experiment run → dvc_data_md5 → exact dataset
               → dvc_model_md5 → exact model artifact
```

---

## DVC: Tradeoffs

**Strengths**
- Exact data lineage across every run
- Works with any remote storage (S3, GCS, Azure)
- Pairs naturally with Git workflows

**Considerations**
- Adds complexity to the workflow (`dvc add`, `dvc push`, `dvc pull`)
- Best suited for **operational workflows** where you run often and need to track data changes alongside config changes
- Lightweight alternative: log data paths and file sizes manually if full DVC integration is overkill

---

## Experiment Tracking: MLflow

MLflow runs **locally** — no account or network required.

```bash
python scripts/train_sst_mlflow.py --run-name "baseline"
mlflow ui --backend-store-uri runs/sst_enso/mlruns
# → http://localhost:5000
```

**What gets tracked per run:**
- Parameters: `n_lags`, `test_size`, `seed`, `roll`
- Metrics: R², RMSE
- Tags: git commit, branch, DVC hashes
- Artifacts: config JSON, predictions CSV, feature importance, plot
- Model: registered as `sst_enso_predictor`

---

## Experiment Tracking: W&B

W&B is **cloud-hosted** — results are accessible from any browser.

```bash
wandb login
python scripts/train_sst_wandb.py --run-name "experiment_1"
# → Run URL printed to console
```

**What gets tracked per run:**
- Same parameters, metrics, and artifacts as MLflow
- Model linked to W&B Model Registry via artifact versioning
- Built-in dashboards: parallel coordinates, scatter plots, run comparison

---

## MLflow vs W&B: Side-by-Side

| | MLflow | W&B |
|---|---|---|
| **Hosting** | Local (file-based) | Cloud (wandb.ai) |
| **Setup** | No account needed | Free account required |
| **UI** | Local — intuitive for solo workflows | Cloud — great for teams |
| **Model registry** | Local or remote (S3, DB) | Cloud; free tier: 5 models |
| **Sharing** | Zip run dir or use shared storage | Share project URL |
| **Offline** | Always offline | `WANDB_MODE=offline`, sync later |
| **CI/CD** | Set `MLFLOW_TRACKING_URI` | Set `WANDB_API_KEY` |

---

## Model Registries

Both tools support registering and loading models by name/version.

**MLflow**
```python
model = mlflow.sklearn.load_model("models:/sst_enso_predictor/latest")
```

**W&B**
```python
artifact = run.use_artifact("wandb-registry-workshop-sst/models:v0.6.2")
model = joblib.load(Path(artifact.download()) / "model.joblib")
```

**W&B free tier limit:** up to 5 models in the registry.
An academic license may remove this limit (application required).

---

## Model Serving with Docker

The trained model is packaged into a Docker image published to GHCR.

```bash
# Pull and run
docker pull ghcr.io/chicago-aiscience/workshop-sst-serve:latest
docker run -p 8000:8000 ghcr.io/chicago-aiscience/workshop-sst-serve:latest

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[0.5, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2]]}'
```

**Anyone with Docker can pull and run the model — no Python environment needed.**

---

## Docker: Using a Custom Model

To swap in your own trained model, mount it as a volume:

```bash
docker run -p 8000:8000 \
  -v /path/to/your/model.joblib:/app/model/model.joblib:ro \
  ghcr.io/chicago-aiscience/workshop-sst-serve:latest
```

**Available endpoints:**

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /model-info` | Feature names and schema |
| `POST /predict` | Run inference |

---

## Other Sharing Options

**HuggingFace Hub**
- Good for open-sourcing models publicly
- Strong discoverability — useful if you want the community to find and use your model
- Model cards encourage documentation

**MLflow on shared storage**
- Zip the `runs/` directory and share, or point to remote storage (Amazon S3, Google Cloud, FTP or SFTP server, NFS, HDFS)
- No extra accounts needed for collaborators who have storage access

**W&B Teams**
- Assign roles (Admin, Member, View-Only) to collaborators
- Set project visibility: private, team, or public

---

## Conclusions

**DVC + experiment trackers** is a powerful combo for reproducibility, but adds workflow complexity. It's best when you need strict data lineage across many runs.

**MLflow** is the better choice for:
- Fully local or offline workflows
- Intuitive solo UI for comparing runs

**W&B** is the better choice for:
- Cloud collaboration and distributed teams
- Rich dashboards and easy sharing via URL

**Docker on GHCR** is the cleanest model distribution story — one command to run, mount your own model to customize.

---

## Recommendations by Use Case

| Use Case | Recommended Stack |
|---|---|
| Solo researcher, local | MLflow + DVC |
| Team collaboration | W&B + DVC |
| Sharing a model publicly | Docker (GHCR) or HuggingFace |
| On-premise | MLflow + local storage on the cluster |
| Strict data lineage | DVC (with either tracker) |

---

## References

- [MLflow documentation](https://mlflow.org/docs/latest/index.html)
- [Weights & Biases documentation](https://docs.wandb.ai/)
- [DVC documentation](https://dvc.org/doc)
- [Model Tracking Guide](model_tracking.md)
- [Model Serving Guide](model_serving.md)
- Image: `ghcr.io/chicago-aiscience/workshop-sst-serve:latest`
