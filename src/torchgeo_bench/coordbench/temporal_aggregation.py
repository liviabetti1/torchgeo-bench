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
    "weekly": 7 * DAY_SECONDS,
    "biweekly": 14 * DAY_SECONDS,
    "monthly": 30 * DAY_SECONDS,
    "seasonally": 90 * DAY_SECONDS,
    "yearly": 365 * DAY_SECONDS,
}

EMBEDDING_AGGREGATION_METHODS = [
    "statistics",
    "temporal_filter",
    "sequence_model",
]

LABEL_STATS = ["mean", "min", "max"]


def dataset_discretization(
    dataset: CoordBenchmark,
    method: str,
    drop_incomplete: bool = True,
) -> CoordBenchmark:
    """Group observations by location and temporal window."""
    t0 = time.time()

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

    t1 = time.time()
    print(f"Discretized dataset in {t1 - t0:.2f} seconds, {len(aggregated)} rows")

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
    batch_size: int = 4096,
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

    for start in tqdm(range(0, len(lats), locations_per_batch), desc="Encoding", unit="batch"):
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
    batch_size: int = 4096,
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

    discretized_dataset = dataset_discretization(dataset, method=discretization_method)

    aggregated_embeddings = _aggregate_embeddings(
        lats=discretized_dataset.lat,
        lons=discretized_dataset.lon,
        timestamps=np.asarray(discretized_dataset.posix_timestamp, dtype=np.int64),
        encoder=encoder,
        batch_size=batch_size,
    )

    return discretized_dataset, aggregated_embeddings


if __name__ == "__main__":
    # EXAMPLE USAGE
    from torchgeo_bench.coordbench.datasets import load_era5_ecmwf
    from torchgeo_bench.coordbench.models import GeoCLIPLocationEncoder

    (era5,) = load_era5_ecmwf()
    discretize_dataset_and_aggregate_embeddings(era5, GeoCLIPLocationEncoder(), 'weekly', 'statistics')
