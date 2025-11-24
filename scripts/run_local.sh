#!/usr/bin/env bash
set -euo pipefail
sst --sst data/sst_sample.csv --enso data/nino34_sample.csv --out-dir artifacts --start 2000-01
echo "Artifacts written to ./artifacts"
