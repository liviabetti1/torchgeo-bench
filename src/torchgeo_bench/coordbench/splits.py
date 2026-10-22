"""Spatial and temporal fold assignment for CoordBench's blockCV-style evaluation splits."""

import numpy as np


def spatial_fold_ids(
    lat: np.ndarray, lon: np.ndarray, folds: int = 5, cell_deg: float = 10.0, seed: int = 0
) -> np.ndarray:
    """Assign each point a fold by its lat/lon grid cell (blockCV-style holdout).

    All points in a ``cell_deg`` block share a fold, so train and test sets are
    spatially disjoint.

    Args:
        lat: Latitudes, shape ``(N,)``.
        lon: Longitudes, shape ``(N,)``.
        folds: Number of folds.
        cell_deg: Grid-cell size in degrees.
        seed: RNG seed for the cell -> fold assignment.

    Returns:
        Per-point fold id in ``[0, folds)``, shape ``(N,)``.
    """
    # deg x deg cells assignments of lat lons
    cell = np.floor(np.asarray(lat) / cell_deg).astype(np.int64) * 100003 + np.floor(
        np.asarray(lon) / cell_deg
    ).astype(np.int64)
    uniq = np.unique(cell)

    # random order the cells
    order = np.random.default_rng(seed).permutation(len(uniq))
    # assigns each cell to a fold
    fold_of = {int(c): int(order[i] % folds) for i, c in enumerate(uniq)}

    # assign each lat,lon to a fold
    return np.array([fold_of[int(c)] for c in cell], dtype=np.int64)


def temporal_fold_ids(
    posix_timestamp: np.ndarray, method: str, folds: int = 5, **kwargs
) -> np.ndarray:
    """Assign each point a fold based on the timestamp.

    Args:
        posix_timestamp: POSIX timestamps, shape ``(N,)``.
        method: Splitting strategy, ``"random"``, ``"grid"``, or ``"forecast"``.
        folds: Number of folds.
        **kwargs: Forwarded to the method-specific helper.

    Returns:
        Per-point fold id in ``[0, folds)``, shape ``(N,)``.
    """
    if method == "random":
        return _random_temporal(posix_timestamp, folds=folds, **kwargs)
    if method == "grid":
        return _grid_temporal(posix_timestamp, folds=folds, **kwargs)
    if method == "forecast":
        return _forecast_temporal(posix_timestamp, folds=folds, **kwargs)
    raise NotImplementedError(f"{method} not implemented yet.")


def _random_temporal(posix_timestamp: np.ndarray, folds: int = 5, seed: int = 42) -> np.ndarray:
    """Assign each point a fold by randomly permuting unique timestamps.
    So, each point remains in a fold, and the timestamps for a given point are not split up across folds.

    Args:
        posix_timestamp: POSIX timestamps, shape ``(N,)``.
        folds: Number of folds.
        seed: Random seed for shuffling timestamps.

    Returns:
        Per-point fold id in ``[0, folds)``, shape ``(N,)``.
    """
    # unique timestamps
    uniq, inverse = np.unique(posix_timestamp, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(uniq))

    # assign each unique timestamp to a fold (randomly wrt order)
    fold_of_uniq = np.empty(len(uniq), dtype=np.int64)
    fold_of_uniq[order] = np.arange(len(uniq)) % folds

    # get fold of each lat lon
    return fold_of_uniq[inverse]


def _grid_temporal(
    posix_timestamp: np.ndarray, temporal_grid_size: float, folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Assign each point a fold by its position in a fixed-size temporal grid.

    Args:
        posix_timestamp: POSIX timestamps, shape ``(N,)``.
        temporal_grid_size: Width of each temporal grid cell (currently in POSIX seconds).
        folds: Number of folds.
        seed: Random seed controlling the offset of the first grid cell.

    Returns:
        Per-point fold id in ``[0, folds)``, shape ``(N,)``.
    """
    # partition timestamps into cells, which are blocks of contiguous time
    cell = np.floor(np.asarray(posix_timestamp) / temporal_grid_size).astype(np.int64)
    uniq, inverse = np.unique(cell, return_inverse=True)

    # to vary which fold is first
    offset = np.random.default_rng(seed).integers(folds)
    # get folds, randomly varying placement
    fold_of_uniq = (uniq - offset) % folds
    return fold_of_uniq[inverse]


def _forecast_temporal(posix_timestamp: np.ndarray, folds: int = 5) -> np.ndarray:
    """Assign each point a fold by how recent it is (forecast-style, walk-forward blocks).

    Timestamps are split into ``folds`` contiguous blocks ordered by time, with
    fold 0 the most recent block.

    Args:
        posix_timestamp: POSIX timestamps, shape ``(N,)``.
        folds: Number of folds.

    Returns:
        Per-point fold id in ``[0, folds)``, shape ``(N,)``.
    """
    uniq, inverse = np.unique(posix_timestamp, return_inverse=True)

    # last one will be 0
    rank_from_latest = (len(uniq) - 1) - np.arange(len(uniq))
    fold_of_uniq = (rank_from_latest * folds) // len(uniq)

    # the lowest fold is held out first
    # but note that after the first fold, the collection of the remaining folds will not be contiguous...
    return fold_of_uniq[inverse]
