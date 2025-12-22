# SST — Sea Surface Temperature & ENSO Prediction

[![status](https://img.shields.io/badge/status-teaching%20demo-blue)](#status)
[![python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](#installation)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![DOI](https://zenodo.org/badge/1087330899.svg)](https://doi.org/10.5281/zenodo.17613101)
[![CI/CD](https://github.com/chicago-aiscience/workshop-sst/actions/workflows/deploy.yml/badge.svg)](https://github.com/chicago-aiscience/workshop-sst/actions/workflows/deploy.yml)

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

**Using `uv` (recommended)**:
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package and dependencies
uv sync --extra dev
```

**Using pip**:
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

## Additional Setup

- Optional: point the CLI at your own SST/ENSO CSV files; sample data already ships in `data/`.
- Linux/Mac users can enable tab completion for `sst` by running `eval "$(uv run sst --install-completion)"` (or `eval "$(sst --install-completion)"` if installed via pip).
- Optional: run code quality checks locally to mirror the CI checks.

```bash
# Using uv (recommended)
uv run ruff check .          # linting
uv run ruff format --check . # formatting check
uv run mypy src/ tests/      # type checking
uv run pyproject-fmt --check pyproject.toml # pyproject.toml formatting

# Or using pip
ruff check . && ruff format --check . && mypy src/ tests/ && pyproject-fmt --check pyproject.toml
```

## Get Started

Run the ML prediction workflow end-to-end:

```bash
# Using uv (recommended)
uv run sst \
  --sst data/sst_sample.csv \
  --enso data/nino34_sample.csv \
  --out-dir artifacts \
  --start 2000-01
```

Or use the default parameters:

```bash
uv run sst
```

**Note**: If you installed via `pip install -e '.[dev]'` and activated the virtual environment, you can use `sst` directly instead of `uv run sst`.

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

## CI/CD

This project uses GitHub Actions for continuous integration and deployment. The workflow (`.github/workflows/deploy.yml`) automatically runs on every push and pull request to `dev` and `main` branches.

### What the CI/CD Pipeline Does

1. **Lint and Format** - Runs ruff, mypy, and pyproject-fmt to ensure code quality and formatting
2. **Security Scanning** - Performs CodeQL analysis to detect security vulnerabilities
3. **Testing** - Runs tests across Python 3.10, 3.11, and 3.12
4. **Version Management** - Automatically calculates next version:
   - `dev` branch: patch + release candidate (e.g., `0.2.0rc1`)
   - `main` branch: minor version (e.g., `0.2.0`)
5. **Docker Build** - Builds and pushes Docker images to GitHub Container Registry
6. **Release** - On successful deployment, creates git tags and GitHub releases

### Version Bumping Strategy

The project follows semantic versioning with automated version bumps:

- **Dev branch** (`dev`):
  - Bumps **patch** version and adds release candidate suffix
  - Example: `0.1.0` → `0.1.1rc1` → `0.1.1rc2` → ...
  - Creates prerelease tags and GitHub releases
  - Used for development and testing

- **Main branch** (`main`):
  - Bumps **minor** version for production releases
  - Example: `0.1.0` → `0.2.0` → `0.3.0` → ...
  - Creates stable release tags and GitHub releases
  - Used for production deployments

**Version Lifecycle**:
1. Version is calculated early in the workflow (before Docker build)
2. Docker images are tagged with the calculated version
3. Version bump is committed to repository only after successful Docker build
4. Git tags and GitHub releases are created with the new version
5. Version bump commits include `[skip ci]` to prevent workflow loops

This ensures the repository version always matches what was actually deployed.

### Viewing CI/CD Status

- Check workflow runs in the [Actions tab](https://github.com/chicago-aiscience/workshop-sst/actions)
- View security scan results in the [Security tab](https://github.com/chicago-aiscience/workshop-sst/security)
- Find Docker images in the [Packages section](https://github.com/chicago-aiscience/workshop-sst/pkgs/container/workshop-sst-sst)

## Developer

This section provides guidance for developers who want to set up, install, run, and contribute to the SST codebase.

### Prerequisites

- **Python**: 3.10 or higher (3.10, 3.11, or 3.12 are supported)
- **Git**: For version control
- **uv** (recommended) or **pip**: Python package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sst
   ```

2. **Install the package and dependencies**:

   **Using `uv` (recommended)**:
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install the package and all development dependencies
   uv sync --extra dev
   ```

   **Using pip**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e '.[dev]'
   ```

  *This installs the package in editable mode along with all development dependencies (testing, linting, formatting, documentation tools).*

### Running the Code

**Run the CLI**:
```bash
# Using uv (recommended)
uv run sst                                    # Using default parameters
uv run sst --sst data/sst_sample.csv --enso data/nino34_sample.csv --out-dir artifacts --start 2000-01

# Using pip (if installed via pip)
sst                                           # Using default parameters
sst --sst data/sst_sample.csv --enso data/nino34_sample.csv --out-dir artifacts --start 2000-01

# Or using the convenience script
./scripts/run_local.sh
```

**Run tests**:
```bash
# Using uv (recommended)
uv run pytest -q              # Quick test run
uv run pytest -v              # Verbose output
uv run pytest tests/          # Run specific test directory

# Using pip
pytest -q              # Quick test run
pytest -v              # Verbose output
pytest tests/          # Run specific test directory
```

### Development Workflow

**Code Quality Checks**:

The project uses several tools to maintain code quality. Run these checks before committing:

**Using `uv` (recommended)**:
```bash
# Lint with ruff
uv run ruff check .

# Format check with ruff
uv run ruff format --check .

# Type checking with mypy
uv run mypy src/ tests/

# Format pyproject.toml
uv run pyproject-fmt --check pyproject.toml

# Or run all checks at once
uv run ruff check . && uv run ruff format --check . && uv run mypy src/ tests/ && uv run pyproject-fmt --check pyproject.toml
```

**Using pip**:
```bash
# Lint with ruff
ruff check .

# Format check with ruff
ruff format --check .

# Type checking with mypy
mypy src/ tests/

# Format pyproject.toml
pyproject-fmt --check pyproject.toml
```

**Project Structure**:
```
sst/
├── src/sst/          # Main package source code
│   ├── cli.py        # Command-line interface
│   ├── io.py         # Data loading functions
│   ├── ml.py         # Machine learning models
│   ├── plot.py       # Visualization functions
│   └── transform.py  # Data transformation utilities
├── tests/            # Test suite
├── data/             # Sample data files
├── docs/             # Documentation source
├── scripts/          # Utility scripts
└── pyproject.toml    # Project configuration and dependencies
```

### Building Documentation

The project uses MkDocs for documentation. To build and serve the documentation locally:

**Using `uv` (recommended)**:
```bash
# Install documentation dependencies
uv sync --extra docs

# Serve documentation locally (with live reload)
uv run mkdocs serve

# Build static documentation site
uv run mkdocs build
```

**Using pip**:
```bash
# Install documentation dependencies
pip install -e '.[docs]'

# Serve documentation locally (with live reload)
mkdocs serve

# Build static documentation site
mkdocs build
```

The documentation will be available at `http://127.0.0.1:8000` when using `mkdocs serve`.

### Docker Development

For containerized development:

```bash
# Build the Docker image
docker build -t sst .

# Run the container
docker run --rm \
  -v "$(pwd)/artifacts":/app/artifacts \
  sst

# Run with custom parameters
docker run --rm \
  -v "$(pwd)/artifacts":/app/artifacts \
  sst --start 2000-01 --n-lags 5

# Run with custom data files
docker run --rm \
  -v "$(pwd)/artifacts":/app/artifacts \
  -v "$(pwd)/my_data":/app/data \
  sst \
  --sst data/my_sst_sample.csv \
  --enso data/my_nino34_sample.csv
```

### Contributing

Before contributing, please:

1. **Fork the repository** and create a feature branch
2. **Set up your development environment** (see Setup above)
3. **Make your changes** following the project's code style
4. **Run tests and quality checks**:
   ```bash
   # Using uv (recommended)
   uv run pytest -q
   uv run ruff check . && uv run ruff format --check . && uv run mypy src/ tests/ && uv run pyproject-fmt --check pyproject.toml

   # Using pip
   pytest -q
   ruff check . && ruff format --check . && mypy src/ tests/ && pyproject-fmt --check pyproject.toml
   ```
5. **Add or update tests** for any behavioral changes
6. **Update documentation** if needed
7. **Submit a pull request** with a clear description

For more detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Development Dependencies

The project includes the following development tools (installed via `uv sync --extra dev` or `pip install -e '.[dev]'`):

- **Testing**: `pytest` - Test framework
- **Linting & Formatting**: `ruff` - Fast Python linter and formatter (replaces Black and Flake8)
- **Type Checking**: `mypy` - Static type checker
- **Configuration Formatting**: `pyproject-fmt` - pyproject.toml formatter
- **Documentation**: `mkdocs`, `mkdocs-material`, `mkdocstrings` - Documentation tools
- **Jupyter**: `jupyterlab` - For notebook development

All development dependencies are defined in `pyproject.toml` under `optional-dependencies.dev`.

## Community

- **Contributing**: Fork this repository, create a feature branch, and open a pull request. Please add or update tests for behavioral changes and run `pytest -q` plus code quality checks (`ruff check .`, `ruff format --check .`, `mypy src/ tests/`, `pyproject-fmt --check pyproject.toml`) before submitting.
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
  sst
```

Adjust the command arguments if you want to point at different input CSVs, change the output directory, or modify ML parameters (e.g., `--n-lags`, `--test-size`).

## Citation

If you use SST in teaching materials or publications, please cite it as:

> Tebaldi, N. (2025). *SST: Sea Surface Temperature & ENSO Prediction* (Version 0.1.0) [Software]. AI+Science Workshops. https://github.com/AI-Science-Workshop/sst

Also consider acknowledging contributors in your workshop materials and linking back to this repository so others can access the resources.
