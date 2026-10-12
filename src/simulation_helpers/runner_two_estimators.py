
"""One-replicate and chunk runners for the two-estimator CI study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.data_generation import generate_two_estimator_data
from utils.estimators import psi_two_estimator, ci_tau_two_estimator


def run_one(params: dict, seed: int | None = None) -> dict:
    """Run one independent replicate and return a flat result row."""
    seed = params["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)

    n1 = params["n1"]
    n2 = params["n2"]
    mu = params["mu"]
    sigma1 = params["sigma1"]
    sigma2 = params["sigma2"]
    rho = params["rho"]
    alpha = params["alpha"]
    gamma = params["gamma"]

    y, _, _, _ = generate_two_estimator_data(
        n1=n1,
        n2=n2,
        mu=mu,
        delta_f=params["delta_f"],
        sigma1=sigma1,
        sigma2=sigma2,
        rho=rho,
        rng=rng,
        return_samples=True,
    )

    eb_hat = y[0] + psi_two_estimator(
        y, n1=n1, n2=n2,
        sigma1=sigma1, sigma2=sigma2, rho=rho,
    ) * (y[1] - y[0])

    regular_hat = y[0]

    ci_eb = ci_tau_two_estimator(
        y, n1=n1, n2=n2,
        sigma1=sigma1, sigma2=sigma2,
        rho=rho, alpha=alpha, gamma=gamma,
    )

    z = norm.ppf(1 - alpha / 2)
    se = sigma1 / np.sqrt(n1)
    ci_regular = (
        regular_hat - z * se,
        regular_hat + z * se,
    )

    row = dict(params)
    row["seed"] = seed

    for name, estimate, ci in [
        ("eb", eb_hat, ci_eb),
        ("regular", regular_hat, ci_regular),
    ]:
        row[f"{name}_estimate"] = float(estimate)
        row[f"{name}_error"] = float(estimate - mu)
        row[f"{name}_squared_error"] = float((estimate - mu) ** 2)
        row[f"{name}_ci_lower"] = float(ci[0])
        row[f"{name}_ci_upper"] = float(ci[1])
        row[f"{name}_coverage"] = int(ci[0] <= mu <= ci[1])
        row[f"{name}_ci_length"] = float(ci[1] - ci[0])

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
