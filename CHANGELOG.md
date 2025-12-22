# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.6.2]

### Added

### Changed
- Migrated to `uv` instead of `pre-commit` for lint and format
- Use `uv` to run all CI/CD steps and execute `sst`

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2025-12-01

### Added

- Initial release of SST (Sea Surface Temperature & ENSO Prediction) package
- Command-line interface (CLI) for running ML prediction workflows
- Support for loading SST and ENSO data from CSV files with sample data included
- Data loading, tidying, transformation, and joining utilities
- 12-month rolling mean smoothing for time series data
- Random Forest model implementation for ENSO prediction from SST data
- Configurable lag features, train-test split, and model persistence to `.joblib` format
- Multi-panel prediction plots with predictions over time, scatter plots, and feature importance visualization
- Output artifacts: `ml_predictions.csv`, `ml_feature_importance.csv`, `ml_predictions.png`, and `model.joblib`
- CLI with default parameters, customizable paths, configurable ML parameters, and tab completion support
- Python package structure with type hints, automated testing with pytest, and pre-commit hooks
- Code quality tools: Black for formatting, Ruff for linting, and mypy for type checking
- GitHub Actions CI/CD workflows and Docker support for containerized execution
- Documentation with MkDocs, comprehensive README, API docs, CONTRIBUTING.md, CODE_OF_CONDUCT.md, and workshop demo notebook
- Unit tests for CLI and ML components with test fixtures and sample data
- Updated project to use `uv` as the recommended package manager for faster dependency installation, with `pip` remaining as an alternative option
- Updated CI/CD workflows to use `uv` for dependency installation
- Complete CI/CD workflow: `.github/workflows/deploy.yml`

---

[0.1.0]: https://github.com/chicago-aiscience/workshop-sst/releases/tag/v0.1.0
