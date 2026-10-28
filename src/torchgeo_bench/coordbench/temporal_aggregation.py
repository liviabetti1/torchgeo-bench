"""Temporal discretization of CoordBenchmark datasets."""

import time
import numpy as np
import pandas as pd
import torch
from rich.progress import track

from torchgeo_bench.coordbench.benchmark import CoordBenchmark
from torchgeo_bench.coordbench.config_enums import TemporalResolution
from torchgeo_bench.coordbench.models import LocationEncoder

GROUP_KEYS = ["lat", "lon", "window_index"]  # keys to group by when discretizing/aggregating over time

# Since Posix timestamps are in seconds can convert days to seconds to specify windows
DAY_SECONDS = 24 * 60 * 60

DISCRETIZATION_METHODS_TO_SECONDS = {
    "daily": DAY_SECONDS,
    "weekly": 7 * DAY_SECONDS,
    "biweekly": 14 * DAY_SECONDS,
    "4_weeks": 28 * DAY_SECONDS,
    "13_weeks": 91 * DAY_SECONDS,
    "days_30": 30 * DAY_SECONDS,
    "days_365": 365 * DAY_SECONDS,
}
# Calendar windows, so their length follows the actual month or (leap) year
CALENDAR_PERIODS = {"monthly": "M", "yearly": "Y"}

# Calendar months covered by each season; "year" keeps every timestamp.
SEASONAL_PERIODS = {
    "year": list(range(1, 13)),
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
    "winter": [12, 1, 2],
}

def _sample_times(start: int, end: int, discretization: str, season: str | None = None) -> np.ndarray:
    """POSIX timestamps in [start, end): one per fixed-length step or calendar period start, optionally restricted to a season."""
    if discretization in CALENDAR_PERIODS:
        starts = pd.date_range(
            pd.to_datetime(start, unit="s"),
            pd.to_datetime(end, unit="s"),
            freq=f"{CALENDAR_PERIODS[discretization]}S",
            inclusive="left",
        )
        times = ((starts - pd.Timestamp(0)) // pd.Timedelta("1s")).to_numpy()
    else:
        times = np.arange(start, end, DISCRETIZATION_METHODS_TO_SECONDS[discretization])

    if season is None:
        return times
    months = pd.to_datetime(times, unit="s").month
    return times[np.isin(months, SEASONAL_PERIODS[season])]


# Statistics each aggregation method concatenates, channel-wise, into the probe features.
POOLED_STATS = {
    "mean": ["mean"],
    "mean_std_slope_tv": ["mean", "std", "slope", "total_variation"],
}
EMBEDDING_AGGREGATION_METHODS = [*POOLED_STATS, "temporal_filter", "sequence_model"]

EMBEDDING_STATS = ["mean", "min", "max", "std", "slope", "total_variation"]
LABEL_STATS = ["mean", "min", "max"]

def pool_embeddings(embeddings: dict[str, torch.Tensor], method: str) -> np.ndarray:
    """Concatenate the statistics `method` pools into probe features of shape (N, len(stats) * D)."""
    if method not in POOLED_STATS:
        raise NotImplementedError(f"Embedding aggregation method not implemented: {method}")
    return torch.cat([embeddings[stat] for stat in POOLED_STATS[method]], dim=1).numpy()


def non_daily_temporal_sampling(dataset: CoordBenchmark, discretization: str, season: str = "year"):
    if dataset.temporal_resolution not in (TemporalResolution.WEEKLY, TemporalResolution.MONTHLY, TemporalResolution.YEARLY):
        raise NotImplementedError("Only do temporal sampling on datasets that are more coarse than daily")

    timestamps_with_leapdays = []
    min_size = 365
    max_size = 0
    for i in track(range(len(dataset.lon)), f"{dataset.name}_temporal_sampling_{discretization}"):
        start = dataset.posix_timestamp[i]
        end = dataset.end_timestamp[i]

        new_timestamps = _sample_times(start, end, discretization, season)
        timestamps_with_leapdays.append(new_timestamps)

        if max_size < new_timestamps.shape[0]:
            max_size = new_timestamps.shape[0]
        
        if min_size > new_timestamps.shape[0]:
            min_size = new_timestamps.shape[0]

    if min_size == 0:
        raise ValueError(
            f"{dataset.name}: {discretization} windows contain no timestamps in season '{season}'"
        )

    # because of leap years, we can run into an inconsistent number of temporal samples from year to year when we're not discretizing by month.  In that case, we remove the last timestamp
    timestamps = []

    samples_to_ignore = []
    if dataset.temporal_resolution != TemporalResolution.WEEKLY:
        for samples in timestamps_with_leapdays:
            leap_year_removed = samples[:min_size]
            timestamps.append(leap_year_removed)

    else: #weekly sampling can vary from 0 to 6 samples, so we have to manually drop all samples that aren't a full week
        for i in range(len(dataset.lon)):
            timestamp = timestamps_with_leapdays[i]
            if len(timestamp) != max_size:
                samples_to_ignore.append(i)
            else:
                timestamps.append(timestamp)

    keep = np.ones(len(dataset.lon), dtype=bool)
    keep[samples_to_ignore] = False

    if len(samples_to_ignore) != 0:
        for task_name, task_values in dataset.tasks.items():
            dataset.tasks[task_name] = task_values[keep]

    spatial_aggregation_key = dataset.spatial_aggregation_key
    if spatial_aggregation_key is not None:
        # Positional reindex: spatial aggregation reads this by .iloc alongside plain arrays.
        column, values = spatial_aggregation_key
        spatial_aggregation_key = (column, values[keep].reset_index(drop=True))

    return CoordBenchmark(
        name=f"{dataset.name}_{season}_{discretization}_discretization",
        lat=dataset.lat[keep],
        lon=dataset.lon[keep],
        tasks=dataset.tasks,
        task_type=dataset.task_type,
        temporal_resolution=dataset.temporal_resolution,
        year=dataset.year,
        posix_timestamp=np.asarray(timestamps),
        spatial_aggregation_key=spatial_aggregation_key,
    )


def no_timestamp_temporal_sampling(
    dataset: CoordBenchmark, season: str, discretization: str
) -> CoordBenchmark:
    """Give every observation of a dataset without timestamps the same grid of timestamps.

    The grid covers `dataset.year`, is spaced by `discretization`, and is restricted to `season`.
    """
    if dataset.year is None:
        raise ValueError("Dataset must have a year to be temporally sampled.")

    if season not in SEASONAL_PERIODS:
        raise ValueError(f"Unknown season: {season}")

    if discretization not in DISCRETIZATION_METHODS_TO_SECONDS and discretization not in CALENDAR_PERIODS:
        raise ValueError(f"Unknown discretization method: {discretization}")

    start = pd.Timestamp(year=dataset.year, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=dataset.year + 1, month=1, day=1, tz="UTC")
    times = _sample_times(int(start.timestamp()), int(end.timestamp()), discretization, season)

    return CoordBenchmark(
        name=f"{dataset.name}_{season}_{discretization}",
        lat=dataset.lat,
        lon=dataset.lon,
        tasks=dataset.tasks,
        task_type=dataset.task_type,
        temporal_resolution=dataset.temporal_resolution,
        year=dataset.year,
        posix_timestamp=np.tile(times, (len(dataset.lat), 1)),
        spatial_aggregation_key=dataset.spatial_aggregation_key,
    )


def dataset_discretization(
    dataset: CoordBenchmark,
    method: str,
    drop_incomplete: bool = True,
) -> CoordBenchmark:
    """Group observations by location and temporal window."""

    if dataset.test_mask is not None:
        raise NotImplementedError("Cannot discretize a dataset with a test mask right now.")

    if dataset.posix_timestamp is None:
        print("No timestamp data for this dataset")
        return dataset

    if method not in DISCRETIZATION_METHODS_TO_SECONDS and method not in CALENDAR_PERIODS:
        raise ValueError(f"Unknown discretization method: {method}")

    task_val_aggregations = ["mode"] if dataset.task_type == "classification" else LABEL_STATS

    timestamp = np.asarray(dataset.posix_timestamp, dtype=np.int64)
    if method in CALENDAR_PERIODS:
        # use pandas periods
        window_index = pd.to_datetime(timestamp, unit="s").to_period(CALENDAR_PERIODS[method]).asi8
        window_index = window_index - window_index.min()
        n_windows = window_index.max() + 1
    else:
        interval = DISCRETIZATION_METHODS_TO_SECONDS[method] # number of seconds in the discretization interval
        elapsed = timestamp - timestamp.min() # time since the first observation in seconds
        span = int(elapsed.max()) + 1   # total number of seconds in the dataset

        if drop_incomplete:
            n_windows = span // interval # will only keep windows that are fully covered by the dataset
        else:
            n_windows = int(np.ceil(span / interval))
        window_index = elapsed // interval # which window each observation belongs to, i.e. an index

    # convert into a dataframe to use pandas groupby for aggregation
    rows = pd.DataFrame({
        "lat": dataset.lat,
        "lon": dataset.lon,
        "window_index": window_index,
        "timestamp": timestamp,
        "row": np.arange(len(timestamp)),
        **dataset.tasks,
    })
    rows = rows[rows["window_index"] < n_windows] # only keep complete windows if drop_incomplete is True
    # group keys are lat, lon, and window_index (i.e. the temporal window)
    rows = rows.sort_values([*GROUP_KEYS, "timestamp"], kind="stable") # sort by lat, lon, window_index, and timestamp to ensure that each group is contiguous
    grouped = rows.groupby(GROUP_KEYS, sort=False) # group by lat, lon, and window_index to aggregate over time

    # check all windows have the same number of timestamps
    window_size = grouped.size()
    if window_size.nunique() != 1:
        raise ValueError("All windows must have the same number of timestamps for now -- might change later.")
    T = int(window_size.iloc[0]) # number of timestamps per window

    # Rows are sorted by (lat, lon, window, timestamp),
    # so each contiguous bloc of T rows corresponds to a single (lat, lon, window) group
    # therefore, we can reshape the timestamps into a 2D array of shape (N, T) where N is the number of unique (lat, lon, window) groups
    timestamps = rows["timestamp"].to_numpy().reshape(-1, T) # (N, T)

    # Aggregate the tasks over time for each (lat, lon, window) group
    # for classification tasks, we use the mode of the labels over time
    # for regression tasks, we use the mean, min, and max of the labels over time
    aggregated = grouped.agg(**{
        f"{task}_{agg}": (task, (lambda s: s.mode().iloc[0]) if agg == "mode" else agg)
        for task in dataset.tasks
        for agg in task_val_aggregations
    }).reset_index()

    return CoordBenchmark(
        name=f"{dataset.name}_{method}",
        lat=aggregated["lat"].to_numpy(),
        lon=aggregated["lon"].to_numpy(),
        tasks={
            f"{task}_{stat}": aggregated[f"{task}_{stat}"].to_numpy()
            for task in dataset.tasks
            for stat in task_val_aggregations
        },
        task_type=dataset.task_type,
        temporal_resolution=TemporalResolution(method),
        year=dataset.year,
        posix_timestamp=timestamps,
        spatial_aggregation_key=dataset.spatial_aggregation_key,
    )

@torch.inference_mode()
def _aggregate_embeddings(
    lats: np.ndarray,
    lons: np.ndarray,
    timestamps: np.ndarray,
    encoder: LocationEncoder,
    batch_size: int = 16384,
) -> dict[str, torch.Tensor]:
    """Encode each location's window of timestamps and "pool" over time.

    Args:
        lats, lons: 1D arrays of length N.
        timestamps: Array of shape (N, T), one window of timestamps per location.
        encoder: Location encoder.
        batch_size: Maximum number of observations per encoder call.

    Returns:
        Mapping of stat name to tensor of shape (N, D).
        Keys: "mean", "min", "max", "std", "slope", "total_variation".
    """
    T = timestamps.shape[1]
    locations_per_batch = max(1, batch_size // T)

    aggregated_embeddings: dict[str, list[torch.Tensor]] = {s: [] for s in EMBEDDING_STATS}

    # Centered time index for OLS slope
    t_index = torch.linspace(-1, 1, T)  # (T,)
    t_var = (t_index**2).sum()

    #for start in tqdm(range(0, len(lats), locations_per_batch), desc="Encoding", unit="batch", leave=False):
    for start in track(range(0, len(lats), locations_per_batch), f"Aggregating embeddings"):
        end = start + locations_per_batch

        batch_lats = lats[start:end]
        batch_lons = lons[start:end]
        batch_timestamps = timestamps[start:end]  # (B, T)

        B, T = batch_timestamps.shape

        # Repeat each location for every timestamp
        batch_lats = np.repeat(batch_lats, T)
        batch_lons = np.repeat(batch_lons, T)
        batch_timestamps = batch_timestamps.reshape(-1)

        # Encode all location-timestamp pairs
        embeddings = encoder.encode(
            batch_lons,
            batch_lats,
            batch_timestamps,
        )  # (B*T, D)

        # Restore temporal dimension
        embeddings = torch.as_tensor(embeddings).reshape(B, T, -1)

        mean_embeddings = embeddings.mean(dim=1)
        min_embeddings = embeddings.amin(dim=1)
        max_embeddings = embeddings.amax(dim=1)
        std_embeddings = embeddings.std(dim=1)

        # Per-channel OLS slope against centered time index
        slope_embeddings = (embeddings * t_index[None, :, None]).sum(dim=1) / t_var

        # Sum of absolute consecutive differences
        total_variation_embeddings = (embeddings[:, 1:] - embeddings[:, :-1]).abs().sum(dim=1)

        expected = (B, embeddings.shape[-1])
        for stat, value in (
            ("mean", mean_embeddings),
            ("min", min_embeddings),
            ("max", max_embeddings),
            ("std", std_embeddings),
            ("slope", slope_embeddings),
            ("total_variation", total_variation_embeddings),
        ):
            assert value.shape == expected, f"{stat}: expected {expected}, got {value.shape}"
            aggregated_embeddings[stat].append(value.cpu())

    aggregated_embeddings = {
        stat: torch.cat(values, dim=0)
        for stat, values in aggregated_embeddings.items()
    }

    return aggregated_embeddings

def discretize_dataset_and_aggregate_embeddings(
    dataset: CoordBenchmark,
    encoder: LocationEncoder,
    discretization_method: str,
    embedding_aggregation_method: str = "mean",
    batch_size: int = 16384,
) -> tuple[CoordBenchmark, tuple[torch.Tensor, ...]]:
    """Generate and aggregate temporal embeddings for a discretized dataset.

    Returns:
        Original dataset and (mean, min, max) embeddings,
        each of shape (N, D).
    """

    if dataset.posix_timestamp is None:
        raise ValueError("Dataset must contain timestamps.")

    if embedding_aggregation_method not in EMBEDDING_AGGREGATION_METHODS:
        #NEED TO IMPLEMENT FOR OTHER METHODS TOO
        raise ValueError(f"Unknown aggregation method: {embedding_aggregation_method}")

    t0 = time.perf_counter()
    discretized_dataset = dataset_discretization(dataset, method=discretization_method)
    t1 = time.perf_counter()
    print(f"Discretized dataset in {t1 - t0:.2f} seconds. {len(discretized_dataset.lat)} locations, {discretized_dataset.posix_timestamp.shape[1]} timestamps per location.")

    aggregated_embeddings = _aggregate_embeddings(
        lats=discretized_dataset.lat,
        lons=discretized_dataset.lon,
        timestamps=np.asarray(discretized_dataset.posix_timestamp, dtype=np.int64),
        encoder=encoder,
        batch_size=batch_size,
    )
    t2 = time.perf_counter()
    print(f"Aggregated embeddings in {t2 - t1:.2f} seconds. {aggregated_embeddings['mean'].shape[0]} locations, {aggregated_embeddings['mean'].shape[1]} embedding dimensions.")

    return discretized_dataset, aggregated_embeddings

def temporal_sample_and_aggregate_embeddings(
    dataset: CoordBenchmark,
    encoder: LocationEncoder,
    season: str,
    discretization_method: str,
    embedding_aggregation_method: str = "mean",
    batch_size: int = 16384,
) -> tuple[CoordBenchmark, tuple[torch.Tensor, ...]]:
    """Generate and aggregate temporal embeddings for a temporally sampled dataset.

    Args:
        season: Key of `SEASONAL_PERIODS`; only timestamps in its months are embedded.

    Returns:
        Original dataset and (mean, min, max) embeddings,
        each of shape (N, D).
    """
    if dataset.posix_timestamp is not None and dataset.temporal_resolution == TemporalResolution.DAILY:
        raise ValueError("This method should only be used on datasets with no timestamps or on datasets with a coarse temporal resolution.")

    if embedding_aggregation_method not in EMBEDDING_AGGREGATION_METHODS:
        raise ValueError(f"Unknown aggregation method: {embedding_aggregation_method}")


    if dataset.temporal_resolution is None:
        temporally_sampled_dataset = no_timestamp_temporal_sampling(
            dataset, season, discretization_method
        )
    else:
        temporally_sampled_dataset = non_daily_temporal_sampling(dataset, discretization_method, season)

    aggregated_embeddings = _aggregate_embeddings(
        lats=temporally_sampled_dataset.lat,
        lons=temporally_sampled_dataset.lon,
        timestamps=np.asarray(temporally_sampled_dataset.posix_timestamp, dtype=np.int64),
        encoder=encoder,
        batch_size=batch_size,
    )

    # need this to be 1D again if there's any spatial pooling downstream
    if len(temporally_sampled_dataset.posix_timestamp.shape) > 1:
        temporally_sampled_dataset.posix_timestamp = temporally_sampled_dataset.posix_timestamp[:,0]

    return temporally_sampled_dataset, aggregated_embeddings


if __name__ == "__main__":
    # EXAMPLE USAGE
    from torchgeo_bench.coordbench.datasets import load_era5_ecmwf, load_chelsa_daily
    from torchgeo_bench.coordbench.models import SinCosLocationEncoder

    (era5,) = load_era5_ecmwf(subsample_locations=False)
    (chelsa,) = load_chelsa_daily(subsample_locations=False)
    discretize_dataset_and_aggregate_embeddings(era5, SinCosLocationEncoder(), 'weekly', 'mean')
    discretize_dataset_and_aggregate_embeddings(chelsa, SinCosLocationEncoder(), 'weekly', 'mean')
