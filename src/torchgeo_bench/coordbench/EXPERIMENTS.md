# Datasets for Benchmarking Temporal SatCLIP

| #  | Dataset             | Temporal info                       | Run script                                                            |
|----|---------------------|-------------------------------------|-----------------------------------------------------------------------|
| 1  | US Trees            | Timestamps                          | `src/torchgeo_bench/coordbench/scripts/run_dm_us_trees.sh`            |
| 2  | SatCLIP Elevation   | None (static)                       |                                                                       |
| 3  | ERA5 / CHELSA daily | Daily, 2017                         |                                                                       |
| 4  | CHELSA monthly      | Monthly, 2011-2018                  | `src/torchgeo_bench/coordbench/scripts/run_chelsa_monthly.sh`         |
| 5  | Electrical load     | Daily, 2016-2023 (individual years) | `src/torchgeo_bench/coordbench/scripts/run_electrical_load_daily.sh`  |
| 6  | Electrical load     | Weekly 2016-2023                    | `src/torchgeo_bench/coordbench/scripts/run_electrical_load_weekly.sh` |
| 7  | USA population      | Yearly, 2010-2025                   | `src/torchgeo_bench/coordbench/scripts/run_usa_census.sh`             |
| 8  | VIIRS Nightlights   | Yearly, 2016-2021                   | `src/torchgeo_bench/coordbench/scripts/run_viirs_night_lights.sh`     |
| 9  | Global Population   | Yearly, 2015-2025                   | `src/torchgeo_bench/coordbench/scripts/run_global_population.sh`      |
| 10 | GHCNd               | ?                                   | **TODO:** still needed                                                |

---

## 1. US Trees

This dataset has timestamps. We can run all models on it to compare performance, although note that not all models use temporal information. For these models, just the lat/lon is fed in. This is to compare whether adding temporal information helps.

**To run:**

```bash
bash src/torchgeo_bench/coordbench/scripts/run_dm-us_trees.sh
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

**Supported resolutions:** `weekly`, `biweekly`, `4_weeks`, `13_weeks`, `days_30`, `days_365`, `monthly`, `yearly`

**Two ways to aggregate:**

1. **Naive timestamp aggregation:** take the average of the timestamps to create datasets at these resolutions.
2. **Embedding aggregation:** take the average of the daily embeddings generated at each day within a window. *(Note: this only really works for encoders that allow daily-resolution temporal encoding.)*

## 4. CHELSA Monthly

A multiyear dataset from 2011-2018, used to test trends over multiple years.  When operating with monthly resolution
datasets, we perform temporal weekly temporal subsampling, then average the results of those embeddings to use for the
whole month.

> `./src/coordbench/scripts/run_chelsa_monthly.sh`

## 5. Electrical Load Daily

Electrical load data in the US from 2016-2023. This is daily data at the county level, split by year (otherwise we run 
into OOM issues). So you will see results for 2016, then 2017, then 2018, etc. Currently defaults to using the centroid 
of each county for speed purposes, but you can turn on spatial aggregation (see below)

> `./src/coordbench/scripts/run_electrical_load_daily.sh`

## 6. Electrical Load Weekly

Electrical load data in the US from 2016-2023.  This is aggregated to the weekly level to benchmark against the totality
of the temporal extent.  As with the electrical load daily dataset, this is at the county-level.  We default to using
the centroid, but can turn on spatial aggregation as needed (see below).  When working with weekly temporal resolution
datasets, we default to temporal subsampling every two days within the week, then average the resulting embeddings.

## 7. USA Population

Yearly USA population data from 2010-2025.  This data is also at the county-level.  We default to the centroid, but can
turn on spatial aggregation as needed.  When working with annual temporal resolution datasets, we default to temporal
subsampling every month, then average the resulting embeddings

> `./src/coordbench/scripts/run_usa_census.sh`

##  8. VIIRS Night Lights
VIIRS Night Lights data from 2016-2021.  This is global, 1 km resolution, annual data; we follow the temporal 
subsampling procedure outlined in USA Population.  Current performance on VIIRS is incredibly poor, to the point that 
this is a low-priority dataset.

> `./src/coordbench/scripts/run_viirs_night_lights.sh`

## 9. Global Population
Global, 1km resolution, annual global population datasets.  We follow the temporal subsampling procedure outlined in
USA Population.  We have a global dataset, and then a dataset for each continent (sans Antarctica).  However, if you are
just interested in the global task, you can adjust the dataloader in `datasets.py` accordingly.

> `./src/coordbench/scripts/run_global_population.sh`

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
2. **Embedding generation and aggregation** — generates an embedding per timestamp and aggregates across the window. `--temporal-embedding-aggregation-methods` picks the probe features: `mean` (default) or `mean_std_slope_tv`, which concatenates the channelwise mean, std, slope, and total variation.

# Spatial Aggregation (for Polygon, Admin-Level Datasets)

We default to using the centroid for polygon-style datasets.  However, we have code available for performing spatial grouping
and aggregation.  This is currently only available for `usa_census`, `usa_electric_load_daily`, and `usa_electric_load_weekly`.

It requires two flags:
- `--from-polygon`: designates that this is a polygon dataset that will be performing some sort of spatial aggregation.  If not present, it is assumed that the centroid will be used.
- `--spatial-aggregation-methods`: designates the aggregation method used on the embeddings of each point within a polygon.  This takes in a list.

**Spatial Aggregation Methods:** `mean`, `statistical`, `median_and_iqr`

# Temporal Splits Experiments

## Nested Cross Validation
Every run is now nested cross-validation, so we do not choose hyperparameters based on test folds. An outer split makes `--folds` train/test folds; an inner (nested) split makes `--inner-folds` tuning folds (train/val) inside each outer train fold, which select the probe's hyperparameters (ridge alpha, `--knn-ks`, `--idw-ks` x `--idw-cs`). The selected setting is refit on the outer train fold and scored on its test fold, and the reported metric is the mean over outer folds.

`--nested-splits` takes `outer,inner` pairs (note, first is outer), e.g. `--nested-splits random,random spatial,random temporal_checkerboard,random`. `--split-test-type official` instead uses the dataset's own `test_mask` as the single outer test set (datasets without one are skipped). Note: I'm not 100% sure, but I think we might want to just use random inner splits for now, but there's an option to try other inner splits

**Split Methods:**

| Splitter | Held out |
|----------|----------|
| `random` | Random rows (KFold) |
| `spatial` | Whole `--cell-deg` grid cells (GroupKFold) |
| `temporal_checkerboard` | Every `folds`-th time unit, rotating across folds |
| `temporal_forecast` | The next `--test-size` units after an expanding training window |
| `temporal_forecast_block` | The next `--test-size` units after a sliding window of `--max-train-size` units |
| `temporal_contiguous_block` | `folds` evenly spaced time units |

Temporal splits skip benchmarks without timestamps.

### `--time-units unit,size`

Defines how timestamps are grouped into **calendar-aligned blocks**. Supported units are `timestamp`, `day`, `week`, `month`, `quarter`, `season`, and `year`. The `size` specifies how many units each block spans.

Examples:
- `week,1`: 1-week blocks
- `week,2`: 2-week blocks
- `day,30`: 30-day blocks
- `year,2`: 2-year blocks

You can specify multiple pairs, and each temporal split runs separately for each:

`--time-units week,1 month,1 day,30`

For `temporal_checkerboard` and `temporal_contiguous_block`, one block is one held-out unit, so `--time-units` alone sets how much time each held-out block covers. For example, `week,2` holds out 2-week windows. For `timestamp`, `size` is ignored and each distinct timestamp is its own block.

Only units that contain data count. If some weeks are missing, `temporal_checkerboard` rotates over the weeks that are present.

### Forecasting: `--test-size` and `--max-train-size`

`--test-size` (default `1`) only applies to `temporal_forecast` and `temporal_forecast_block` (with any other split it will be 1). This sets how many time units each fold forecasts. `--max-train-size` sets how many units the sliding training window of `temporal_forecast_block` holds. Both count `--time-units` blocks. For example, to train on 5 weeks and forecast the next 2, use `--time-units week,1 --max-train-size 5 --test-size 2`.

```bash
bash src/torchgeo_bench/coordbench/scripts/run_era5_temporal_split.sh --device cuda:0
```

`--save-predictions` writes the out-of-fold predictions of every split to `.parquet` files alongside the labels.

# Avoiding Antarctica
CLIMPLICIT doesn't have pre-training data in Antarctica and does incredibly poorly there as a result.  To facilitate
'fair' comparisons, points in Antarctica are filtered out of our benchmark tasks by default.  This only affects global
tasks, such as VIIRS night lights, CHELSA, ERA5, etc.  Pass `--no-exclude-antarctica` to keep them.

# Summary of flags now

| Flag                                       | Description                                                                                                                                                                                                 |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--split-test-type`                        | `nested_cv` (default) or `official` (use the dataset's test mask as the outer test set)                                                                                                                     |
| `--nested-splits`                          | `outer,inner` splitter pairs, e.g. `random,random spatial,random temporal_checkerboard,random`                                                                                                              |
| `--folds` / `--inner-folds`                | Number of outer / inner (tuning) folds                                                                                                                                                                      |
| `--time-units`                             | `unit,size` pairs for temporal splits, e.g. `week,1 month,1`; each temporal split runs once per pair                                                                                                        |
| `--test-size` / `--max-train-size`         | Time units forecast per fold (`temporal_forecast` splits only, default 1) / in the sliding training window (`temporal_forecast_block`)                                                                       |
| `--reset-year`                             | Time units do not cross a year boundary                                                                                                                                                                     |
| `--methods`                                | Any of `linear`, `knn` (classification only), `idw` (spatiotemporal inverse-distance weighting on coordinates)                                                                                             |
| `--knn-ks` / `--idw-ks` / `--idw-cs`       | Hyperparameter grids tuned in the inner CV; `--idw-cs` weights time distance against space distance                                                                                                        |
| `--temporal-discretization-methods`        | Window sizes for datasets that already have timestamps (e.g. `weekly`, `biweekly`, `4_weeks`, `13_weeks`)                                                                                                   |
| `--temporal-sampling-methods`              | Time frame to sample for static datasets (`year`, `spring`, `summer`, `fall`, `winter`)                                                                                                                     |
| `--temporal-sampling-year`                 | Calendar year to sample timestamps from (required for static datasets)                                                                                                                                      |
| `--temporal-embedding-aggregation-methods` | How to pool per-timestamp embeddings into one vector: `mean` (default) or `mean_std_slope_tv`                                                                                                               |
| `--skip-no-timestamps`                     | Skip benchmarks that have no native timestamps (useful when only temporal models are being evaluated)                                                                                                       |
| `--from-polygon`                           | Designates that the dataset(s) are being generated from a polygon, and therefore need some form of spatial aggregation (which happens after temporal aggregation, if both are present)                      |
| `--spatial-aggregation-methods`            | Sets the spatial aggregation method used.  Options are `mean`, `statistical`, and `median_and_iqr`                                                                                                          |
| `--subsample`                              | If applicable, will run on a 'toy' subsampled version of the dataset; useful for iterating quickly.                                                                                                         |
| `--save-predictions`                       | Saves .parquet files of your labels and out-of-fold predictions in your results.  Useful for visualizing data in the plot_fine_grained_model_predictions.ipynb notebook                                     |
| `--no-exclude-antarctica`                  | Keeps rows in Antarctica, which are filtered out of your benchmark dataset(s) by default.                                                                                                                   |