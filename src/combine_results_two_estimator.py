
"""Combine two-estimator EB simulation results.

Usage:
    python src/combine_results_two_estimator.py --input results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results"))
    args = parser.parse_args()

    input_dir = args.input
    files = sorted(input_dir.glob("results_chunk_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No chunk files found in {input_dir}")

    print(f"Found {len(files)} chunk files.")

    df = pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
    )

    # Separate successful and failed replicates.
    if "error" in df.columns:
        failed = df[df["error"].notna()].copy()
        df = df[df["error"].isna()].copy()
    else:
        failed = pd.DataFrame()

    print(f"Successful replicates: {len(df)}")
    print(f"Failed replicates: {len(failed)}")

    if not failed.empty:
        failed.to_parquet(input_dir / "failed_results.parquet", index=False)

    if df.empty:
        raise ValueError("No successful replicates to combine.")

    df = df.sort_values(["scenario_id", "rep"]).reset_index(drop=True)
    df.to_parquet(input_dir / "combined_results.parquet", index=False)

    # Aggregate performance metrics by scenario.
    group_cols = [
        "scenario_id", "n1", "n2", "mu", "delta_f",
        "sigma1", "sigma2", "rho", "alpha", "gamma",
    ]

    summary = df.groupby(group_cols).agg(
        B=("rep", "count"),
        mse_eb=("eb_squared_error", "mean"),
        mse_regular=("regular_squared_error", "mean"),
        coverage_eb=("eb_coverage", "mean"),
        coverage_regular=("regular_coverage", "mean"),
        length_eb=("eb_ci_length", "mean"),
        length_regular=("regular_ci_length", "mean"),
    ).reset_index()

    summary["mse_ratio"] = summary["mse_eb"] / summary["mse_regular"]
    summary["length_ratio"] = summary["length_eb"] / summary["length_regular"]

    summary = summary.sort_values("scenario_id")
    summary.to_parquet(input_dir / "summary_results.parquet", index=False)

    print(f"Saved {input_dir / 'combined_results.parquet'}")
    print(f"Saved {input_dir / 'summary_results.parquet'}")
    print(f"Scenarios: {len(summary)}")


if __name__ == "__main__":
    main()
