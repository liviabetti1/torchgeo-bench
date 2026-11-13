"""Row partitioning for CoordBench: splitters and the official / nested-CV evaluation protocols."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    PredefinedSplit,
    TimeSeriesSplit,
)

from torchgeo_bench.coordbench.benchmark import CoordBenchmark
from torchgeo_bench.coordbench.config import CoordEvaluationConfig

SPLITS = (
    "random",
    "spatial",
    "temporal_forecast",
    "temporal_checkerboard",
    "temporal_forecast_block",
    "temporal_contiguous_block",
)
TIME_UNITS = ("timestamp", "day", "week", "month", "quarter", "season", "year")


@dataclass(frozen=True)
class OuterFold:
    """One outer train/test split and the tuning pairs inside its train rows."""

    train: np.ndarray  # global row ids
    test: np.ndarray
    inner: list[tuple[np.ndarray, np.ndarray]]  # (train, val) for each inner fold, all inside `train`

def spatial_cell_ids(
    lat: np.ndarray, lon: np.ndarray, cell_deg: float = 10.0
) -> np.ndarray:
    """Returns a unique cell id for each lat/lon pair, based on a grid of ``cell_deg`` x ``cell_deg`` cells.
    """
    # deg x deg cells assignments of lat lons
    cell = np.floor(np.asarray(lat) / cell_deg).astype(np.int64) * 100003 + np.floor(
        np.asarray(lon) / cell_deg
    ).astype(np.int64)
    return cell

# maybe want to use this for temporal aggregation too...
def time_unit_ids(
    timestamp: np.ndarray,
    unit_str: str,
    size: int = 1,
    *,
    reset_year: bool = False,
) -> np.ndarray:
    """Assign timestamps to non-overlapping time buckets.

    Args:
        timestamp: Unix timestamps in seconds (UTC).
        unit_str: One of timestamp, day, week, month, quarter,
            season, or year.
        size: Number of units per bucket.
        reset_year: If True, restart buckets at each calendar year
            (except for year units). If False, anchor buckets to
            the unit containing the earliest timestamp.

    Returns:
        Integer bucket ID for each timestamp.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if unit_str == "timestamp":
        return np.asarray(timestamp)

    seconds = np.floor(np.asarray(timestamp, dtype=np.float64)).astype(np.int64).astype("datetime64[s]")
    days = seconds.astype("datetime64[D]").astype(np.int64)  # days since 1970-01-01
    months = seconds.astype("datetime64[M]").astype(np.int64)  # months since 1970-01
    year = months // 12 + 1970
    day_of_year = days - seconds.astype("datetime64[Y]").astype("datetime64[D]").astype(np.int64)

    if unit_str == "day":
        ids = year * 400 + day_of_year // size if reset_year else (days - days.min()) // size
    elif unit_str == "week":
        # NOT CALENDER ALIGNED, so not Monday - Sunday
        weeks = 7 * size
        ids = year * 54 + day_of_year // weeks if reset_year else (days - days.min()) // weeks
    elif unit_str == "month":
        ids = year * 12 + (months % 12) // size if reset_year else (months - months.min()) // size
    elif unit_str == "quarter":
        quarters = months // 3
        ids = year * 4 + (quarters % 4) // size if reset_year else (quarters - quarters.min()) // size
    elif unit_str == "season":
        # use meteorological seasons:, DJF, MAM, JJA, SON
        # if used within a single year, this will be a bit weird
        if reset_year:
            raise ValueError("Meteorological seasons can cross year boundaries.")
        seasons = (months + 1) // 3  # December starts the next year's winter
        ids = (seasons - seasons.min()) // size
    elif unit_str == "year":
        ids = (year - year.min()) // size
    else:
        raise ValueError(f"Unknown time unit {unit_str}")

    return ids.astype(np.int64)


def split(
    cfg: CoordEvaluationConfig,
    split: str,
    folds: int, # number of outer OR inner folds
    bench: CoordBenchmark,
    rows: np.ndarray,
    seed: int,
    time_unit: tuple[str, int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) pairs over ``rows`` for one splitter, as global row ids."""
    if split == "random":
        groups = None
    elif split == "spatial":
        groups = spatial_cell_ids(bench.lat[rows], bench.lon[rows], cfg.cell_deg)
    elif split.startswith("temporal"):
        groups = time_unit_ids(bench.posix_timestamp[rows], *time_unit, reset_year=cfg.reset_year)

    match split:
        case "random":
            splitter = KFold(folds, shuffle=True, random_state=seed)
            pairs = [(rows[tr], rows[te]) for tr, te in splitter.split(rows)]

        case "spatial":
            splitter = GroupKFold(folds, shuffle=True, random_state=seed)
            pairs = [(rows[tr], rows[te]) for tr, te in splitter.split(rows, groups=groups)]

        case "temporal_checkerboard":
            _, inverse = np.unique(groups, return_inverse=True)
            # Consecutive units take turns across folds.
            splitter = PredefinedSplit(inverse % folds)
            pairs = [(rows[tr], rows[te]) for tr, te in splitter.split(rows)]

        case "temporal_forecast" | "temporal_forecast_block":
            # Expanding or sliding window, always forecasting forward.
            units, inverse = np.unique(groups, return_inverse=True)

            splitter = TimeSeriesSplit(
                n_splits=folds,
                max_train_size=cfg.max_train_size if split == "temporal_forecast_block" else None,
                test_size=cfg.test_size,
            )
            pairs = [
                (rows[np.isin(inverse, tr)], rows[np.isin(inverse, te)])
                for tr, te in splitter.split(units)
            ]

        case "temporal_contiguous_block":
            # Hold out `folds` evenly spaced time units.
            units, inverse = np.unique(groups, return_inverse=True)
            if folds > len(units):
                raise ValueError(f"{folds} blocks need {folds} time units, found {len(units)}")
            starts = np.linspace(0, len(units) - 1, folds).astype(int)
            held_out = [inverse == s for s in starts]
            pairs = [(rows[~te], rows[te]) for te in held_out]

        case _:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")

    if any(len(tr) == 0 or len(te) == 0 for tr, te in pairs):
        raise ValueError(f"split={split!r} produced an empty train or test fold on {bench.name}")
    return pairs


def split_variants(cfg: CoordEvaluationConfig) -> list[tuple[str, str, tuple[str, int] | None]]:
    """(outer, inner, time_unit) combinations; pairs without a temporal split use only the first time unit."""
    time_units = cfg.time_units or [None]
    return [
        (outer, inner, unit)
        for outer, inner in cfg.nested_splits
        for unit in (
            time_units
            if outer.startswith("temporal") or inner.startswith("temporal")
            else time_units[:1]
        )
    ]


def make_folds(
    cfg: CoordEvaluationConfig,
    bench: CoordBenchmark,
    rows: np.ndarray,
    seed: int,
    variant: tuple[str, str, tuple[str, int] | None],
) -> list[OuterFold]:
    """Outer folds with inner tuning pairs for one ``split_variants`` entry.

    official: bench.test_mask defines the outer test set; inner CV tunes on the rest.
    nested_cv: each outer split defines an outer CV; inner CV tunes within each outer train.
    """
    outer, inner, unit = variant
    if cfg.split_test_type == "official":
        if bench.test_mask is None:
            raise ValueError(f"split_test_type='official' needs a test_mask; {bench.name} has none")
        is_test = np.asarray(bench.test_mask, dtype=bool)[rows]
        pairs = [(rows[~is_test], rows[is_test])]
    elif cfg.split_test_type == "nested_cv":
        pairs = split(cfg, outer, cfg.folds, bench, rows, seed, unit)
    else:
        raise ValueError(f"Unknown split_test_type: {cfg.split_test_type}")
    return [
        OuterFold(train=tr, test=te, inner=split(cfg, inner, cfg.inner_folds, bench, tr, seed, unit))
        for tr, te in pairs
    ]