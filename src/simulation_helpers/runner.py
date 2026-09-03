"""One-replicate and chunk runners for the simulation study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_generation import generate_data
from utils.estimators import eb, ppi, ppi_plus_plus, ppi_plus_plus_lrt, ppi_plus_plus_oracle


def run_one(params: dict, seed: int | None = None) -> dict:
    """Run one independent replicate and return a flat result row."""
    seed = params["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    data_keys = ("n1", "n2", "mu", "delta_f", "delta_c", "sigma1", "sigma2", "sigma3", "rho")
    y, cov_y, y12, y3 = generate_data(
        rng=rng, return_samples=True, **{key: params[key] for key in data_keys}
    )
    eb_hat = eb(y, params["n1"], params["n2"], params["sigma1"], params["sigma2"], params["sigma3"], params["rho"])
    ppi_hat = ppi(y)
    pp_hat = ppi_plus_plus(y, params["n1"], params["n2"], cov_y)
    estimates = {"eb": eb_hat, "ppi": ppi_hat, "ppi_plus_plus": pp_hat}
    if params.get("ppi_plus_plus_lrt", False):
        estimates["ppi_plus_plus_lrt"] = ppi_plus_plus_lrt(
            y,
            params["n1"],
            params["n2"],
            cov_y,
            y12[:, 1],
            y3,
            params["mu"],
            params["delta_f"],
            params["delta_c"],
            params["sigma2"],
            params["sigma3"],
        )
    if params.get("ppi_plus_plus_oracle", False):
        estimates["ppi_plus_plus_oracle"] = ppi_plus_plus_oracle(
            y,
            params["n1"],
            params["n2"],
            params["sigma1"],
            params["sigma2"],
            params["sigma3"],
            params["rho"],
            params["delta_f"],
            params["delta_c"],
        )
    truth = float(params["mu"])
    row = dict(params)
    row["seed"] = seed
    for name, estimate in estimates.items():
        row[f"{name}_estimate"] = estimate
        row[f"{name}_error"] = estimate - truth
    return row


def run_task_chunk(chunk_id: int, tasks: list[dict], output_dir: str) -> str:
    """Run a chunk, retaining an error row for any failed replicate."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        try:
            rows.append(run_one(task))
        except Exception as exc:
            rows.append({**task, "error": repr(exc)})
    path = output / f"results_chunk_{chunk_id:04d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path)
