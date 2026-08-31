"""Combine chunk CSVs and summarize Monte Carlo performance."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    files = sorted(args.input.glob("results_chunk_*.csv"))
    if not files:
        raise FileNotFoundError(f"No chunk files found in {args.input}")
    raw = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    raw_path = args.output or args.input / "replicate_results.csv"
    raw.to_csv(raw_path, index=False)
    methods = ["eb", "ppi", "ppi_plus_plus"]
    failed = raw["error"].notna() if "error" in raw else pd.Series(False, index=raw.index)
    usable = raw.loc[~failed].copy()
    if usable.empty:
        raise ValueError("All simulation replicates failed; no summary can be computed.")

    # Parameter values identify a scenario.  Do not group by scenario_id:
    # older result files assigned a different ID to every replicate.
    scenario_cols = [
        c for c in ["n1", "n2", "mu", "delta_f", "delta_c", "rho",
                    "sigma1", "sigma2", "sigma3"] if c in raw
    ]
    rows = []
    for keys, all_group in raw.groupby(scenario_cols, dropna=False, sort=True):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = dict(zip(scenario_cols, keys))
        group = all_group.loc[~failed.loc[all_group.index]]
        row["n_replicates"] = len(group)
        row["n_failed"] = int(failed.loc[all_group.index].sum())
        if "scenario_id" in all_group and all_group["scenario_id"].nunique() == 1:
            row["scenario_id"] = all_group["scenario_id"].iloc[0]
        for method in methods:
            estimates = pd.to_numeric(group[f"{method}_estimate"], errors="coerce")
            errors = estimates - pd.to_numeric(group["mu"], errors="coerce")
            row[f"{method}_bias"] = errors.mean()
            row[f"{method}_variance"] = errors.var(ddof=1)
            row[f"{method}_mse"] = (errors ** 2).mean()
        rows.append(row)
    summary_path = args.input / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Wrote {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()
