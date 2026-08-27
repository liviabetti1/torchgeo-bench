import numpy as np

from dataclasses import replace

from torchgeo_bench.coordbench.datasets import CoordBenchmark


def temporal_split(dataset: CoordBenchmark, method: str, **kwargs) -> CoordBenchmark:
    """Split a coordinate benchmark into train/test sets along the temporal axis.

    Args:
        dataset: Coordinate benchmark to split.
        method: Splitting strategy, ``"random"``, ``"grid"``, or ``"forecast"``.
        **kwargs

    Returns:
        A new `CoordBenchmark` with ``test_mask``
    """
    if method == "random":
        return _random_temporal(dataset, **kwargs)
    elif method == "grid":
        return _grid_temporal(dataset, **kwargs)
    elif method == "forecast":
        return _forecast_temporal(dataset, **kwargs)
    else:
        raise NotImplementedError(f"{method} not implemented yet.")


def _random_temporal(dataset: CoordBenchmark, test_size: float = 0.2, seed: int = 42) -> CoordBenchmark:
    """Randomly hold out whole timestamps as the test set.

    Args:
        dataset: Coordinate benchmark to split.
        test_size: Target fraction to hold out for testing.
        seed: Random seed for shuffling timestamps.

    Returns:
        A new `CoordBenchmark` with ``test_mask``
    """
    uniq, inverse, counts = np.unique(dataset.posix_timestamp, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(uniq))

    # Running sample count before each shuffled timestamp is added, so vectorized greedy fill
    running_before = np.cumsum(counts[order]) - counts[order]

    is_test_timestamp = np.zeros(len(uniq), dtype=bool)
    is_test_timestamp[order] = running_before < test_size * len(dataset.posix_timestamp)

    return replace(dataset, test_mask=is_test_timestamp[inverse])


def _grid_temporal(dataset: CoordBenchmark, temporal_grid_size: float, test_size: float = 0.2, seed: int = 42) -> CoordBenchmark:
    """Evenly space held-out test cells across a fixed-size temporal grid.

    Args:
        dataset: Coordinate benchmark to split.
        temporal_grid_size: Width of each temporal grid cell (currently in POSIX seconds)
        test_size: Target fraction hold out for testing
        seed: Random seed controlling offset at beginning for held-out cells.

    Returns:
        A new `CoordBenchmark` with ``test_mask`` 
    """
    cell = np.floor(np.asarray(dataset.posix_timestamp) / temporal_grid_size).astype(np.int64)
    uniq, inverse = np.unique(cell, return_inverse=True)

    stride = round(1 / test_size)  # How often to hold out a test cell
    offset = np.random.default_rng(seed).integers(stride)
    is_test_timestamp = (uniq - offset) % stride == 0

    return replace(dataset, test_mask=is_test_timestamp[inverse])


def _forecast_temporal(dataset: CoordBenchmark, test_size: float = 0.2) -> CoordBenchmark:
    """Hold out the most recent timestamps as the test set (forecast-style split).

    Args:
        dataset: Coordinate benchmark to split.
        test_size: Target fraction to hold out for testing.

    Returns:
        A new `CoordBenchmark` with ``test_mask``
    """
    _, inverse, counts = np.unique(dataset.posix_timestamp, return_inverse=True, return_counts=True)

    is_test_timestamp = np.cumsum(counts[::-1])[::-1] <= test_size * len(dataset.posix_timestamp)

    return replace(dataset, test_mask=is_test_timestamp[inverse])
