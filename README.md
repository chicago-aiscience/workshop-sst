# SST — Sea Surface Temperature & ENSO Prediction

[![status](https://img.shields.io/badge/status-teaching%20demo-blue)](#status)
[![python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](#installation)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![DOI](https://zenodo.org/badge/1087330899.svg)](https://doi.org/10.5281/zenodo.17613101)

SST is a lightweight Python package and CLI (published as `sst`) that demonstrates machine learning prediction of ENSO from Sea Surface Temperature data. It is designed as a workshop to serve as a quick example of ML workflows using monthly Sea Surface Temperature (SST) and ENSO (Niño 3.4) data. The project emphasizes repeatable pipelines, documentation, testing, and CI/CD habits.

## Status
This project is maintained as a teaching demo. Expect stability in the CLI and dataset formats, with occasional updates to support workshops and best-practice examples.

## Features
- Load curated SST and ENSO CSV samples from `data/`
- Tidy, join, and smooth values with 12-month rolling means
- Machine learning prediction of ENSO from SST using Random Forest with lag features
- Produce artifacts: predictions, feature importance, and visualization plots
- Ship with typing, automated tests, and GitHub Actions integration for demonstration

## Installation
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

## Additional Setup

- Optional: point the CLI at your own SST/ENSO CSV files; sample data already ships in `data/`.
- Linux/Mac users can enable tab completion for `sst` by running `eval "$(sst --install-completion)"`.
- Optional: install pre-commit hooks to mirror the CI checks.

```bash
pre-commit install  # enable local formatting and linting checks
pre-commit run --all-files  # run pre-commit on all files
```

## Get Started

Run the ML prediction workflow end-to-end:

```bash
sst \
  --sst data/sst_sample.csv \
  --enso data/nino34_sample.csv \
  --out-dir artifacts \
  --start 2000-01
```

Or use the default parameters:

```bash
sst
```

The CLI generates the following artifacts in the output directory:
- `ml_predictions.csv` - Predictions with actual vs predicted values and residuals
- `ml_feature_importance.csv` - Feature importance scores for each input feature
- `ml_predictions.png` - Multi-panel visualization showing predictions over time, scatter plot, and feature importance
- `model.joblib` - Trained Random Forest model (can be loaded for future predictions)

The CLI completes in a few seconds on a laptop.

## Tests

Run tests:

```bash
pytest -q
```

## Community

- **Contributing**: Fork this repository, create a feature branch, and open a pull request. Please add or update tests for behavioral changes and run `pytest -q` plus `pre-commit run --all-files` before submitting.
- **Discussions & issues**: Use GitHub Issues to request features, report bugs, or share teaching ideas. We welcome suggestions that improve clarity for workshop audiences.
- **Code of conduct**: Be respectful and inclusive. Follow the Python Software Foundation's Code of Conduct spirit in discussions and reviews.
- **Teaching ideas**: Try adding seasonal plots, enforcing schema validation, or practicing release workflows (e.g., tagging `v0.1.0` and attaching artifacts).

## Docker Usage

Build and run the SST ML prediction pipeline inside a container:

```bash
# Build the image locally (once)
docker build -t sst .

# Execute the predict command, writing outputs to ./artifacts on your host
docker run --rm \
  -v "$(pwd)/artifacts":/app/artifacts \
  sst predict
```

Adjust the command arguments if you want to point at different input CSVs, change the output directory, or modify ML parameters (e.g., `--n-lags`, `--test-size`).

## Citation

If you use SST in teaching materials or publications, please cite it as:

> Tebaldi, N. (2025). *SST: Sea Surface Temperature & ENSO Prediction* (Version 0.1.0) [Software]. AI+Science Workshops. https://github.com/AI-Science-Workshop/sst

Also consider acknowledging contributors in your workshop materials and linking back to this repository so others can access the resources.
