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
    rows = []
    group_cols = [c for c in ["n1", "n2", "mu", "delta_f", "delta_c", "rho", "sigma1", "sigma2", "sigma3"] if c in raw]
    for keys, group in raw[raw.get("error").isna() if "error" in raw else slice(None)].groupby(group_cols, dropna=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        base = dict(zip(group_cols, keys))
        for method in methods:
            errors = group[f"{method}_error"]
            rows.append({**base, "method": method, "bias": errors.mean(), "variance": errors.var(ddof=1), "mse": (errors ** 2).mean()})
    summary_path = args.input / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Wrote {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()
