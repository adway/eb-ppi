# EB/PPI simulation study

The simulation is configured in JSON. Scalar fields define one setting; list-valued fields are crossed to form a grid. `n_2` is derived as `n_2_multiplier * n_1`, and `delta` is used for both `delta_f` and `delta_c` as requested.

Set an integer `seed` for reproducible runs. You can also omit `seed` or set it to `null` to generate fresh seeds from system entropy.

From the project root:

```bash
python src/run_simulation.py --config configs/example.json --local --jobs 4
python src/combine_results.py --input results/example
```

For Slurm, set `slurm.account` and submit with:

```bash
python src/run_simulation.py --config configs/example.json --slurm
```

Alternatively, submit from the Slurm wrappers after setting the account in
the `#SBATCH` headers:

```bash
sbatch slurm/deploy_simulation.sh configs/example.json
sbatch slurm/combine_results.sh results/example
```

Each replicate has a deterministic independent seed. Submitit jobs write checkpoint files named `results_chunk_XXXX.csv`, so completed chunks remain usable if a job needs to be retried. The combined replicate-level data and grouped Monte Carlo summary (bias, Monte Carlo variance, and MSE) are written by `combine_results.py`.
