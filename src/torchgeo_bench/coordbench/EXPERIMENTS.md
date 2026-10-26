# Datasets for Benchmarking Temporal SatCLIP

| # | Dataset | Temporal info | Run script |
|---|---------|---------------|------------|
| 1 | US Trees | Timestamps | `src/torchgeo_bench/coordbench/scripts/run_dm_us_trees.sh` |
| 2 | SatCLIP Elevation | None (static) | |
| 3 | ERA5 / CHELSA daily | Daily, 2017 | |
| 4 | CHELSA monthly | Monthly, 2011-2018 | **TODO:** still needed |
| 5 | Electrical load | Daily, 2016-2023 | **TODO:** still needed |
| 6 | USA population | Yearly, 2010-2025 | **TODO:** still needed |
| 7 | VIIRS Nightlights | ?| **TODO:** still needed |
| 8 | GHCNd | ? | **TODO:** still needed |

---

## Setting up your environment
You'll need to export paths for:
- your virtual environment
- your local torchgeo repository
- your local satclip repository

For example, when running on rails, I do the following:
```bash
export TORCHGEO_HOME=/u/kplane19/torchgeo-bench
export TORCHGEO_ENV=/u/kplane19/torchgeo-bench/.venv
export SATCLIP_HOME=/u/kplane19/satclip
export RESULTS_HOME=/projects/bgtj/t_satclip/results
export CKPT_HOME=/projects/bgtj/t_satclip/ckpts
```

The variable names must be TORCHGEO_HOME, TORCHGEO_ENV, SATCLIP_HOME, RESULTS_HOME, and CKPT_HOME for compatibility with our scripts.  For simplicity's sake, you can put these lines in the bottom of your .bashrc file (found inside your /u/username/ directory).  Upon login, they will automatically be added to your environment.  These can (and should) vary from machine to machine as a way to avoid hard-coding relative paths. 

If you are writing your own sbatch .job script, you MUST have the line #sbatch --export=ALL on in order for the job to pick up your local environment variables.

## 1. US Trees

This dataset has timestamps. We can run all models on it to compare performance, although note that not all models use temporal information. For these models, just the lat/lon is fed in. This is to compare whether adding temporal information helps.

**To run:**

```bash
bash src/torchgeo_bench/coordbench/scripts/run_dm_us_trees.sh
```

## 2. SatCLIP Elevation

This dataset is elevation data without timestamps. We can evaluate on it to see if adding temporal information helps even with static tasks.

- **Models without temporal information:** just get lat/lon (the standard way).
- **Models with temporal information:** we can aggregate the embeddings for different times at a given lat/lon, to see if this "composite" is more informative.

**Aggregation methods:**

| Method | What it does |
|--------|--------------|
| `annual` | TODO: detail |
| `summer` | TODO: detail |
| `default_date_winter` | TODO: detail |
| `default_date_summer` | TODO: detail |
| `concat_four_seasons` | TODO: detail |

## 3. ERA5 / CHELSA Daily

These datasets were constructed manually at 30K locations for each day in 2017. They were built so that they can be aggregated at different temporal resolutions.

**Supported resolutions:** `weekly`, `biweekly`, `4_weeks`, `13_weeks`, `monthly`, `seasonally`, `yearly`

**Two ways to aggregate:**

1. **Naive timestamp aggregation:** take the average of the timestamps to create datasets at these resolutions.
2. **Embedding aggregation:** take the average of the daily embeddings generated at each day within a window. *(Note: this only really works for encoders that allow daily-resolution temporal encoding.)*

## 4. CHELSA Monthly

A multiyear dataset from 2011-2018, used to test trends over multiple years.

> **TODO:** still need a bash script for this.

## 5. Electrical Load

Electrical load data in the US from 2016-2023. This is daily data at the county level. We use spatial aggregation for this: we sample 100 points per county and have different spatial aggregation methods.

> **TODO:** still need a bash script for this.

## 6. USA Population

Yearly USA population data from 2010-2025.

> **TODO:** still need a bash script for this.

# Temporal Aggregation Experiments

We have two categories of experiments, depending on whether the dataset carries timestamps.

## Datasets with timestamps

Timestamps are discretized into windows of a chosen resolution (e.g. `weekly`, `biweekly`). Each location's samples that fall within a window are then aggregated into a single embedding.

## Datasets without timestamps (static)

Timestamps are *sampled* for each location across a specified time frame (`year`, `spring`, `summer`, `fall`, `winter`). The same discretization resolutions apply, which controls how densely points are sampled within that frame.

Note that for this, you have to specify `--temporal-sampling-year` (since some models with time input require year...)

## Implementation

We now have two steps to this process:

1. **Discretization / temporal sampling** — produces a dataset where each row is a `(lat, lon, time_window)` tuple.
2. **Embedding generation and aggregation** — generates an embedding per timestamp and aggregates across the window. The default aggregation is *statistical*, which returns channelwise mean, max, and min; `run.py` currently uses only the mean.

# Temporal Split Experiments

`--split temporal` cross-validates over time blocks, so each fold holds out whole stretches of time. It is skipped for benchmarks without timestamps. `--temporal-grid-size` (or `--temporal-grid-sizes` for a sweep) sets the block width and is required.

**Grid sizes:** `daily`, `weekly`, `biweekly`, `4_weeks`, `13_weeks`, `monthly`, `seasonally`, `yearly`

```bash
bash src/torchgeo_bench/coordbench/scripts/run_era5_temporal_split.sh --device cuda:0
```

## Summary of flags now

| Flag | Description |
|------|-------------|
| `--split temporal` | Cross-validate over time blocks (needs a grid size) |
| `--temporal-grid-size(s)` | Width of the held-out time blocks; `--temporal-grid-sizes` sweeps several |
| `--temporal-discretization-methods` | Window sizes for datasets that already have timestamps (e.g. `weekly`, `biweekly`, `4_weeks`, `13_weeks`) |
| `--temporal-sampling-methods` | Time frame to sample for static datasets (`year`, `spring`, `summer`, `fall`, `winter`) |
| `--temporal-sampling-year` | Calendar year to sample timestamps from (required for static datasets) |
| `--temporal-embedding-aggregation-methods` | How to pool per-timestamp embeddings into one vector (default: `statistics`, which computes channelwise mean, min, and max; only the mean is used downstream) |
| `--skip-no-timestamps` | Skip benchmarks that have no native timestamps (useful when only temporal models are being evaluated) |
