# syntax=docker/dockerfile:1.6

FROM python:3.12-slim AS base

# Optimize container build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies required for manylinux wheels (pandas/numpy).
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy source code and project files
COPY pyproject.toml README.md ./
COPY src ./src

# Copy sample data and scripts used by the CLI.
COPY data ./data
COPY scripts ./scripts

# Install project.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

# Default artifacts directory (mounted volume can override).
VOLUME ["/app/artifacts"]

# Execute the container
ENTRYPOINT ["python", "-m", "sst.cli"]
CMD ["--sst", "data/sst_sample.csv", "--enso", "data/nino34_sample.csv", "--out-dir", "artifacts", "--start", "2000-01"]
