#!/bin/bash
# Submit the two-estimator EB confidence interval simulation through Submitit.
# Usage: sbatch slurm/deploy_simulation.sh [config.json]
# Override account with: sbatch --account=... ...

#SBATCH --job-name=eb_ci_submit
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=10:00

set -euo pipefail

CONFIG="${1:-configs/example_two_estimators.json}"

PROJECT_ROOT="$HOME/eb-ppi"
cd "$PROJECT_ROOT"
mkdir -p logs run_logs results

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "Project: $(pwd)"
echo "Config: $CONFIG"

uv sync
echo "Python: $(uv run python --version)"

uv run python -u src/run_simulation_two_estimators.py \
    --config "$CONFIG" \
    --slurm

echo "Submitit jobs submitted."
