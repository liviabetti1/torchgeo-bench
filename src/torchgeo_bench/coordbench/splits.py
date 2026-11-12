"""Row partitioning for CoordBench: splitters and the official / nested-CV evaluation protocols."""

from dataclasses import dataclass
from datetime import UTC, datetime

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
TIME_UNITS = ("timestamp", "day", "week", "month", "season", "year")


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
    reset_year: bool = True,
) -> np.ndarray:
    """Calendar-unit IDs; optionally reset windows at each year boundary."""
    if size < 1:
        raise ValueError("size must be >= 1")
    if unit_str == "timestamp":
        return np.asarray(timestamp)

    dt = [datetime.fromtimestamp(t, UTC) for t in timestamp]

    if unit_str == "day":
        ids = [
            d.year * 400 + (d.timetuple().tm_yday - 1) // size
            if reset_year
            else (d.toordinal() - 1) // size
            for d in dt
        ]
    elif unit_str == "week":
        ids = [
            d.year * 54 + (d.timetuple().tm_yday - 1) // (7 * size)
            if reset_year
            else (d.toordinal() - 1) // (7 * size)
            for d in dt
        ]
    elif unit_str == "month":
        ids = [
            d.year * 12 + (d.month - 1) // size
            if reset_year
            else (d.year * 12 + d.month - 1) // size
            for d in dt
        ]
    elif unit_str == "season":
        ids = [
            d.year * 4 + ((d.month - 1) // 3) // size
            if reset_year
            else (d.year * 4 + (d.month - 1) // 3) // size
            for d in dt
        ]
    elif unit_str == "year":
        ids = [d.year // size for d in dt]
    else:
        raise ValueError(f"Unknown time unit {unit_str}")

    return np.asarray(ids, dtype=np.int64)


def split(
    cfg: CoordEvaluationConfig,
    split: str,
    folds: int, # number of outer OR inner folds
    bench: CoordBenchmark,
    rows: np.ndarray,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) pairs over ``rows`` for one splitter, as global row ids."""
    if split == "random":
        groups = None
    elif split == "spatial":
        groups = spatial_cell_ids(bench.lat[rows], bench.lon[rows], cfg.cell_deg)
    elif split.startswith("temporal"):
        groups = time_unit_ids(bench.posix_timestamp[rows], cfg.time_unit, cfg.time_unit_size, reset_year=cfg.reset_year)

    match split:
        case "random":
            splitter = KFold(folds, shuffle=True, random_state=seed)
            pairs = [(rows[tr], rows[te]) for tr, te in splitter.split(rows)]

        case "spatial":
            splitter = GroupKFold(folds, shuffle=True, random_state=seed)
            pairs = [(rows[tr], rows[te]) for tr, te in splitter.split(rows, groups=groups)]

        case "temporal_checkerboard":
            _, inverse = np.unique(groups, return_inverse=True)
            # Runs of `test_size` consecutive units take turns across folds.
            splitter = PredefinedSplit((inverse // cfg.test_size) % folds)
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
            # Hold out `folds` evenly spaced blocks of `test_size` contiguous calendar units.
            units, inverse = np.unique(groups, return_inverse=True)
            size = cfg.test_size
            if folds * size > len(units):
                raise ValueError(f"{folds} blocks of {size} units need {folds * size} units, found {len(units)}")
            starts = np.linspace(0, len(units) - size, folds).astype(int)
            held_out = [(inverse >= s) & (inverse < s + size) for s in starts]
            pairs = [(rows[~te], rows[te]) for te in held_out]

        case _:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")

    if any(len(tr) == 0 or len(te) == 0 for tr, te in pairs):
        raise ValueError(f"split={split!r} produced an empty train or test fold on {bench.name}")
    return pairs


def make_folds(
    cfg: CoordEvaluationConfig, bench: CoordBenchmark, rows: np.ndarray, seed: int
) -> list[OuterFold]:
    """Outer folds with inner tuning pairs.

    official: bench.test_mask defines the outer test set; inner CV tunes on the rest.
    nested_cv: cfg.split defines outer CV; inner CV tunes within each outer train.
    In both protocols, cfg.inner_split chooses the inner splitter independently of cfg.split.
    """
    if cfg.split_test_type == "official":
        if bench.test_mask is None:
            raise ValueError(f"split_test_type='official' needs a test_mask; {bench.name} has none")
        is_test = np.asarray(bench.test_mask, dtype=bool)[rows]
        outer = [(rows[~is_test], rows[is_test])]

    elif cfg.split_test_type == "nested_cv":
        outer = split(cfg, cfg.split, cfg.folds, bench, rows, seed)

    else:
        raise ValueError(f"Unknown split_test_type: {cfg.split_test_type}")

    return [
        OuterFold(
            train=tr,
            test=te,
            inner=split(cfg, cfg.inner_split, cfg.inner_folds, bench, tr, seed),
        )
        for tr, te in outer
    ]