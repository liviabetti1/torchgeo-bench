"""Temporal discretization of CoordBenchmark datasets."""

import numpy as np
import pandas as pd

from torchgeo_bench.coordbench.benchmark import CoordBenchmark


DISCRETIZATION_METHODS = {
    "weekly": 7 * 24 * 60 * 60,
    "biweekly": 14 * 24 * 60 * 60,
    "monthly": 30 * 24 * 60 * 60,
    "seasonally": 90 * 24 * 60 * 60,
    "yearly": 365 * 24 * 60 * 60,
}

EMBEDDING_AGGREGATION_METHODS = (
    "statistics",
    "temporal_filter",
    "sequence_model",
)


def _make_timestamp_range(
    timestamp_start: int,
    timestamp_end: int,
    timestamp_interval: int,
    drop_incomplete: bool = True,
) -> pd.DatetimeIndex:
    """Return boundaries of non-overlapping temporal windows."""

    time_intervals = pd.date_range(
        start=pd.to_datetime(timestamp_start, unit="s", utc=True),
        end=pd.to_datetime(timestamp_end, unit="s", utc=True),
        freq=pd.Timedelta(seconds=timestamp_interval),
    )

    if not drop_incomplete and time_intervals[-1].timestamp() < timestamp_end:
        time_intervals = time_intervals.append(
            pd.DatetimeIndex([
                pd.to_datetime(timestamp_end, unit="s", utc=True)
            ])
        )

    return time_intervals


def _group_by_location_and_window(
    dataset: CoordBenchmark,
    windows: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """Group original dataset indices by (latitude, longitude, window)."""

    indices = np.flatnonzero(valid)
    indices = indices[np.lexsort((
        windows[indices], dataset.lon[indices], dataset.lat[indices]
    ))]

    groups = {}

    for i in indices:
        key = (dataset.lat[i], dataset.lon[i], windows[i])

        if key not in groups:
            groups[key] = []

        groups[key].append(i)

    return groups


def _aggregate_groups(
    dataset: CoordBenchmark,
    groups: dict,
) -> tuple:
    """Aggregate timestamps and task statistics for each group."""

    lat = []
    lon = []
    timestamps = []
    tasks = {name: [] for name in dataset.tasks}

    for (latitude, longitude, window), group_indices in groups.items():
        lat.append(latitude)
        lon.append(longitude)

        timestamps.append(tuple(dataset.posix_timestamp[group_indices]))

        for name, values in dataset.tasks.items():
            group_values = np.asarray(values)[group_indices]

            tasks[name].append({
                "mean": np.mean(group_values),
                "max": np.max(group_values),
                "min": np.min(group_values),
            })

    new_tasks = {
        f"{name}_{stat}": np.array([group[stat] for group in groups])
        for name, groups in tasks.items()
        for stat in ("mean", "max", "min")
    }

    new_timestamps = np.empty(len(timestamps), dtype=object)
    new_timestamps[:] = timestamps

    return (
        np.asarray(lat),
        np.asarray(lon),
        new_timestamps,
        new_tasks
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

    if method not in DISCRETIZATION_METHODS:
        raise ValueError(f"Unknown discretization method: {method}")

    ts = np.asarray(dataset.posix_timestamp, dtype=np.int64)

    start, end = int(ts.min()), int(ts.max()) + 1
    interval = DISCRETIZATION_METHODS[method]

    time_intervals = (
        _make_timestamp_range(start, end, interval, drop_incomplete)
        .astype("int64") // 10**9
    )

    # Assign each observation to a window
    windows = np.searchsorted(time_intervals, ts, side="right") - 1
    valid = (windows >= 0) & (windows < len(time_intervals) - 1)

    # Group observations by location and window
    groups = _group_by_location_and_window(dataset, windows, valid)

    # Aggregate observations
    lat, lon, timestamps, tasks = _aggregate_groups(dataset, groups)

    return CoordBenchmark(
        name=f"{dataset.name}_{method}",
        lat=lat,
        lon=lon,
        tasks=tasks,
        task_type=dataset.task_type,
        temporal_resolution=method,
        year=dataset.year,
        posix_timestamp=timestamps,
        test_mask=(
            dataset.test_mask[first]
            if dataset.test_mask is not None
            else None
        ),
        spatial_aggregation_key=dataset.spatial_aggregation_key,
    )


def _generate_embeddings(
    lat: np.ndarray,
    lon: np.ndarray,
    timestamps: np.ndarray,
    encoder: LocationEncoder,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Generate embeddings for multiple (lat, lon, timestamp) observations.

    Args:
        lat, lon, timestamps: 1D arrays of equal length.
        encoder: Location encoder.
        batch_size: Maximum number of observations per encoder call.

    Returns:
        Tensor of shape (M, D).
    """
    all_embeddings = []

    for start in range(0, len(lat), batch_size):
        end = start + batch_size

        embeddings = encoder.encode(
            lon[start:end],
            lat[start:end],
            timestamps[start:end],
        )

        all_embeddings.append(torch.as_tensor(embeddings))

    return torch.cat(all_embeddings, dim=0)


def _aggregate_embeddings(
    embeddings: torch.Tensor,
    method: str = "statistics",
) -> tuple[torch.Tensor, ...]:
    """Aggregate temporal embeddings of shape (T, D)."""

    if method == "statistics":
        return {
            "mean": embeddings.mean(dim=0),
            "min": embeddings.min(dim=0).values,
            "max": embeddings.max(dim=0).values,
        }

    elif method == "temporal_filter":
        raise NotImplementedError("Temporal filtering not implemented yet.")

    elif method == "sequence_model":
        raise NotImplementedError("Sequence model not implemented yet.")

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def discretize_dataset_and_aggregate_embeddings(
    dataset: CoordBenchmark,
    encoder: LocationEncoder,
    method: str = "statistics",
    batch_size: int = 4096,
) -> tuple[CoordBenchmark, tuple[torch.Tensor, ...]]:
    """Generate and aggregate temporal embeddings for a discretized dataset.

    Returns:
        Original dataset and (mean, min, max) embeddings,
        each of shape (N, D).
    """

    if dataset.posix_timestamp is None:
        raise ValueError("Dataset must contain timestamps.")

    if method not in EMBEDDING_AGGREGATION_METHODS:
        raise ValueError(f"Unknown aggregation method: {method}")

    # Number of timestamps per location
    lengths = np.array([len(ts) for ts in dataset.posix_timestamp])

    if np.any(lengths == 0):
        raise ValueError("Each location must have at least one timestamp.")

    # Flatten all observations into individual (lat, lon, timestamp) rows
    lat = np.repeat(dataset.lat, lengths)
    lon = np.repeat(dataset.lon, lengths)
    timestamps = np.concatenate(dataset.posix_timestamp)

    # Encode in batches
    embeddings = _generate_embeddings(
        lat=lat,
        lon=lon,
        timestamps=timestamps,
        encoder=encoder,
        batch_size=batch_size,
    )

    # Split embeddings back into temporal groups
    splits = np.cumsum(lengths)[:-1]
    groups = torch.tensor_split(embeddings, splits.tolist())

    # Aggregate each group's embeddings
    aggregated = [
        _aggregate_embeddings(group, method=method)
        for group in groups
    ]

    results = {
        name: torch.stack([group[name] for group in aggregated])
        for name in aggregated[0]
    }

    return dataset, results