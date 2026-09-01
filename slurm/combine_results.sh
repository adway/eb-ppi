#!/bin/bash
# Combine completed simulation chunks into replicate and summary Parquet files.
# Usage: sbatch slurm/combine_results.sh [results-directory]
#SBATCH --job-name=eb_ppi_combine
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=10:00

set -euo pipefail

INPUT="${1:-results}"

PROJECT_ROOT="$HOME/eb-ppi"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "Project: $(pwd)"
echo "Input: $INPUT"
echo "Python: $(uv run python --version)"

uv sync
uv run python -u src/combine_results.py --input "$INPUT"

echo "Results combined."
