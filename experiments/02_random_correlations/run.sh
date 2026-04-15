#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python3 experiments/02_random_correlations/run.py "$@"
