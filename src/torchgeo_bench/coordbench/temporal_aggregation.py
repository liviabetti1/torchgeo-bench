"""Temporal discretization of CoordBenchmark datasets."""

import time
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from torchgeo_bench.coordbench.benchmark import CoordBenchmark
from torchgeo_bench.coordbench.models import LocationEncoder

GROUP_KEYS = ["lat", "lon", "window"]

DAY_SECONDS = 24 * 60 * 60

DISCRETIZATION_METHODS_TO_SECONDS = {
    "daily": DAY_SECONDS,
    "weekly": 7 * DAY_SECONDS,
    "biweekly": 14 * DAY_SECONDS,
    "monthly": 30 * DAY_SECONDS,
    "seasonally": 90 * DAY_SECONDS,
    "yearly": 365 * DAY_SECONDS,
}

# Calendar months of `dataset.year` covered by each window
# To try to capture seasons
TEMPORAL_SAMPLING_MONTHS = {
    "year": list(range(1, 13)),
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
    "winter": [12, 1, 2],
}
TEMPORAL_SAMPLING_METHODS = list(TEMPORAL_SAMPLING_MONTHS)

EMBEDDING_AGGREGATION_METHODS = [
    "statistics",
    "temporal_filter",
    "sequence_model",
]

LABEL_STATS = ["mean", "min", "max"]

def dataset_temporal_sampling(
    dataset: CoordBenchmark, method: str, discretization: str
) -> CoordBenchmark:
    """Assign each observation a regular grid of timestamps over a window of `dataset.year`.

    Args:
        dataset: Benchmark with a year and no timestamps.
        method: Window of the year, a key of `TEMPORAL_SAMPLING_MONTHS`.
        discretization: Step between timestamps, a key of `DISCRETIZATION_METHODS_TO_SECONDS`.
    """
    if dataset.year is None:
        raise ValueError("Dataset must have a year to be temporally sampled.")
    if method not in TEMPORAL_SAMPLING_MONTHS:
        raise ValueError(f"Unknown temporal sampling method: {method}")
    if discretization not in DISCRETIZATION_METHODS_TO_SECONDS:
        raise ValueError(f"Unknown discretization method: {discretization}")
    step = DISCRETIZATION_METHODS_TO_SECONDS[discretization]
    times = []
    for month in TEMPORAL_SAMPLING_MONTHS[method]:
        start = pd.Timestamp(year=dataset.year, month=month, day=1, tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        times.append(np.arange(int(start.timestamp()), int(end.timestamp()), step))
    times = np.concatenate(times)
    return CoordBenchmark(
        name=f"{dataset.name}_{method}_{discretization}",
        lat=dataset.lat,
        lon=dataset.lon,
        tasks=dataset.tasks,
        task_type=dataset.task_type,
        temporal_resolution=f"{method}/{discretization}",
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

    if method not in DISCRETIZATION_METHODS_TO_SECONDS:
        raise ValueError(f"Unknown discretization method: {method}")

    task_val_aggregations = ["mode"] if dataset.task_type == "classification" else LABEL_STATS

    interval = DISCRETIZATION_METHODS_TO_SECONDS[method]
    timestamp = np.asarray(dataset.posix_timestamp, dtype=np.int64)
    elapsed = timestamp - timestamp.min()
    span = int(elapsed.max()) + 1

    if drop_incomplete:
        n_windows = span // interval
    else:
        n_windows = int(np.ceil(span / interval))

    rows = pd.DataFrame({
        "lat": dataset.lat,
        "lon": dataset.lon,
        "window": elapsed // interval,
        "timestamp": timestamp,
        "row": np.arange(len(timestamp)),
        **dataset.tasks,
    })
    rows = rows[rows["window"] < n_windows]
    rows = rows.sort_values([*GROUP_KEYS, "timestamp"], kind="stable")
    grouped = rows.groupby(GROUP_KEYS, sort=False)

    window_size = grouped.size()
    if window_size.nunique() != 1:
        raise ValueError("All windows must have the same number of timestamps.")
    T = int(window_size.iloc[0])

    # Rows are sorted by (lat, lon, window, timestamp), so each consecutive run of T is one group.
    timestamps = rows["timestamp"].to_numpy().reshape(-1, T) # (N, T)

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
        temporal_resolution=method,
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
    """Encode each location's window of timestamps and reduce over time.

    Args:
        lats, lons: 1D arrays of length N.
        timestamps: Array of shape (N, T), one window of timestamps per location.
        encoder: Location encoder.
        batch_size: Maximum number of observations per encoder call.

    Returns:
        Mapping of "mean", "min" and "max" to tensors of shape (N, D).
    """
    T = timestamps.shape[1]
    locations_per_batch = max(1, batch_size // T)

    aggregated_embeddings = {"mean": [], "min": [], "max": []}

    #for start in tqdm(range(0, len(lats), locations_per_batch), desc="Encoding", unit="batch", leave=False):
    for start in range(0, len(lats), locations_per_batch):
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

        assert mean_embeddings.shape == min_embeddings.shape == max_embeddings.shape == (B, embeddings.shape[-1]), (
            f"Expected shape {(B, embeddings.shape[-1])}, got mean {mean_embeddings.shape}, min {min_embeddings.shape}, max {max_embeddings.shape}"
        )

        for stat, value in (
            ("mean", mean_embeddings),
            ("min", min_embeddings),
            ("max", max_embeddings),
        ):
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
    embedding_aggregation_method: str = "statistics",
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
    temporal_sampling_method: str,
    discretization_method: str,
    embedding_aggregation_method: str = "statistics",
    batch_size: int = 16384,
) -> tuple[CoordBenchmark, tuple[torch.Tensor, ...]]:
    """Generate and aggregate temporal embeddings for a temporally sampled dataset.

    Returns:
        Original dataset and (mean, min, max) embeddings,
        each of shape (N, D).
    """

    if dataset.posix_timestamp is not None:
        raise ValueError("This method should only be used on datasets with no timestamps.")

    if embedding_aggregation_method not in EMBEDDING_AGGREGATION_METHODS:
        raise ValueError(f"Unknown aggregation method: {embedding_aggregation_method}")

    temporally_sampled_dataset = dataset_temporal_sampling(
        dataset, temporal_sampling_method, discretization_method
    )
    aggregated_embeddings = _aggregate_embeddings(
        lats=temporally_sampled_dataset.lat,
        lons=temporally_sampled_dataset.lon,
        timestamps=np.asarray(temporally_sampled_dataset.posix_timestamp, dtype=np.int64),
        encoder=encoder,
        batch_size=batch_size,
    )

    return temporally_sampled_dataset, aggregated_embeddings


if __name__ == "__main__":
    # EXAMPLE USAGE
    from torchgeo_bench.coordbench.datasets import load_era5_ecmwf
    from torchgeo_bench.coordbench.models import GeoCLIPLocationEncoder

    (era5,) = load_era5_ecmwf()
    discretize_dataset_and_aggregate_embeddings(era5, GeoCLIPLocationEncoder(), 'weekly', 'statistics')
