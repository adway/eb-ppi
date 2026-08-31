#!/bin/bash
# Submit the configured Monte Carlo chunks through Submitit.
# Usage: sbatch slurm/deploy_simulation.sh [config.json]
# Set the account below (or override it with: sbatch --account=... ...).
#SBATCH --job-name=eb_ppi_submit
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/example.json}"

cd "$PROJECT_ROOT"
mkdir -p logs run_logs results

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "Project: $(pwd)"
echo "Config: $CONFIG"
echo "Python: $(uv run python --version)"

uv sync
uv run python -u src/run_simulation.py --config "$CONFIG" --slurm

echo "Submitit jobs submitted."
