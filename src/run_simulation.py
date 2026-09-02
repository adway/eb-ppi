"""Run a configured simulation locally or through Submitit.

Examples:
    python src/run_simulation.py --config configs/example.json --local
    python src/run_simulation.py --config configs/example.json --slurm
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import submitit

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from simulation_helpers.runner import run_task_chunk


def _values(value):
    return value if isinstance(value, list) else [value]


def load_tasks(path: Path) -> tuple[dict, list[dict]]:
    """Load JSON and expand list-valued fields into a Cartesian parameter grid."""
    config = json.loads(path.read_text())
    required = {"n_1", "n_2_multiplier", "mu", "delta", "rho", "sigma1", "sigma2", "sigma3", "replicates", "slurm"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    fields = ["n_1", "n_2_multiplier", "mu", "delta", "rho", "sigma1", "sigma2", "sigma3"]
    tasks = []
    for scenario_id, values in enumerate(itertools.product(*[_values(config[f]) for f in fields])):
        setting = dict(zip(fields, values))
        n1 = int(setting.pop("n_1"))
        multiplier = setting.pop("n_2_multiplier")
        n2_float = n1 * float(multiplier)
        if n2_float != int(n2_float):
            raise ValueError(f"n_2_multiplier={multiplier} gives noninteger n2 for n1={n1}")
        delta = setting.pop("delta")
        setting.update({"n1": n1, "n2": int(n2_float), "delta_f": delta, "delta_c": delta})
        for rep in range(int(config["replicates"])):
            task = dict(setting, rep=rep)
            task["scenario_id"] = scenario_id  # shared by all replicates in this scenario
            tasks.append(task)
    configured_seed = config.get("seed")
    if configured_seed is None:
        # SeedSequence obtains entropy from the operating system. Each task
        # still receives its own seed, so task order does not share RNG state.
        seeds = np.random.default_rng().integers(0, 2**63 - 1, size=len(tasks), dtype=np.int64)
    else:
        base_seed = int(configured_seed)
        seeds = base_seed + np.arange(len(tasks), dtype=np.int64)
    for task, seed in zip(tasks, seeds):
        task["seed"] = int(seed)
        task["ppi_plus_plus_oracle"] = bool(config.get("ppi_plus_plus_oracle", False))
    return config, tasks


def split_chunks(tasks, n_chunks):
    chunks = [[] for _ in range(max(1, min(n_chunks, len(tasks))))]
    for i, task in enumerate(tasks):
        chunks[i % len(chunks)].append(task)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local", action="store_true", help="run chunks in this process")
    parser.add_argument("--slurm", action="store_true", help="submit chunks with Submitit")
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()
    if args.local == args.slurm:
        parser.error("choose exactly one of --local or --slurm")
    config, tasks = load_tasks(args.config)
    output_dir = ROOT / config.get("output_dir", "results")
    log_dir = ROOT / config.get("log_dir", "run_logs")
    n_jobs = args.jobs or int(config.get("jobs", 1))
    chunks = split_chunks(tasks, n_jobs)
    if args.local:
        for chunk_id, chunk in enumerate(chunks):
            run_task_chunk(chunk_id, chunk, str(output_dir))
        print(f"Completed {len(tasks)} replicates in {len(chunks)} local chunks.")
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(log_dir))
    slurm = config["slurm"]
    slurm_required = {"job_name", "partition", "account", "time_minutes", "memory", "cpus_per_task"}
    missing_slurm = slurm_required - slurm.keys()
    if missing_slurm:
        raise ValueError(f"Missing config fields under slurm: {sorted(missing_slurm)}")
    executor.update_parameters(
        slurm_job_name=slurm["job_name"],
        slurm_partition=slurm["partition"],
        slurm_account=slurm["account"],
        slurm_time=slurm["time_minutes"],
        slurm_mem=slurm["memory"],
        cpus_per_task=slurm["cpus_per_task"],
        tasks_per_node=1,
        slurm_setup=[f"export PYTHONPATH={SRC}:$PYTHONPATH", f"cd {ROOT}"],
    )
    with executor.batch():
        jobs = [executor.submit(run_task_chunk, i, chunk, str(output_dir)) for i, chunk in enumerate(chunks)]
    print(f"Submitted {len(jobs)} jobs for {len(tasks)} replicates.")
    print([job.job_id for job in jobs])


if __name__ == "__main__":
    main()
