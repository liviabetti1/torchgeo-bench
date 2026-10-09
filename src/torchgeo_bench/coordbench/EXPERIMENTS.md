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

**Supported resolutions:** `1_week`, `2_week`, `4_week`, `13_week`, `annual`

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
