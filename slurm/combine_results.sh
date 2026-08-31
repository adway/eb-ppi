#!/bin/bash
# Combine completed simulation chunks into replicate and summary CSVs.
# Usage: sbatch slurm/combine_results.sh [results-directory]
#SBATCH --job-name=eb_ppi_combine
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-results/example}"

cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "Project: $(pwd)"
echo "Input: $INPUT"
echo "Python: $(uv run python --version)"

uv sync
uv run python -u src/combine_results.py --input "$INPUT"

echo "Results combined."
