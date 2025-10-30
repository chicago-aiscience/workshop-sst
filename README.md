# SST ETL — Sea Surface Temperature & ENSO

[![status](https://img.shields.io/badge/status-teaching%20demo-blue)](#status)
[![python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](#installation)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

SST ETL is a lightweight Python package and CLI (published as `sst`) that demonstrates production-minded extract-transform-load practices for climate time-series. It is designed for educators, workshop facilitators, and Python newcomers who want a realistic-yet-fast example of data engineering workflows using monthly Sea Surface Temperature (SST) and ENSO (Niño 3.4) data. The project emphasizes repeatable pipelines, documentation, testing, and CI/CD habits.

## Status
This project is maintained as a teaching demo. Expect stability in the CLI and dataset formats, with occasional updates to support workshops and best-practice examples.

## Features
- Load curated SST and ENSO CSV samples from `data/`
- Tidy, join, and smooth values with 12-month rolling means
- Calculate core metrics (decadal trend, year-over-year changes, correlations)
- Produce publication-ready artifacts: `artifacts/summary.csv` and `artifacts/trends.png`
- Ship with typing, automated tests, and GitHub Actions integration for demonstration

## Installation
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

## Additional Setup
- Optional: install pre-commit hooks to mirror the CI checks.
- Optional: point the CLI at your own SST/ENSO CSV files; sample data already ships in `data/`.
- Linux/Mac users can enable tab completion for `sst` by running `eval "$(sst --install-completion)"`.

```bash
pre-commit install  # enable local formatting and linting checks
pre-commit run --all-files  # run pre-commit on all files
```

## Get Started

Run the full ETL workflow end-to-end and inspect outputs:

```bash
sst \
  --sst data/sst_sample.csv \
  --enso data/nino34_sample.csv \
  --out-dir artifacts \
  --start 2000-01
```

Run tests:

```bash
pytest -q
```

Artifacts land in `artifacts/summary.csv` and `artifacts/trends.png`. The CLI completes in a few seconds on a laptop.

## Community
- **Contributing**: Fork this repository, create a feature branch, and open a pull request. Please add or update tests for behavioral changes and run `pytest -q` plus `pre-commit run --all-files` before submitting.
- **Discussions & issues**: Use GitHub Issues to request features, report bugs, or share teaching ideas. We welcome suggestions that improve clarity for workshop audiences.
- **Code of conduct**: Be respectful and inclusive. Follow the Python Software Foundation's Code of Conduct spirit in discussions and reviews.
- **Teaching ideas**: Try adding seasonal plots, enforcing schema validation, or practicing release workflows (e.g., tagging `v0.1.0` and attaching artifacts).

## Citation
If you use SST ETL in teaching materials or publications, please cite it as:

> Tebaldi, N. (2025). *SST ETL: Sea Surface Temperature & ENSO pipeline* (Version 0.1.0) [Software]. AI+Science Workshops. https://github.com/AI-Science-Workshop/sst

Also consider acknowledging contributors in your workshop materials and linking back to this repository so others can access the resources.
