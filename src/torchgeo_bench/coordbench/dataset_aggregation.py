"""Temporal aggregation utilities for coordinate benchmarks.

Coordinate benchmarks with daily-resolution timestamps can be aggregated
into coarser temporal periods (e.g. weekly or multi-week windows) so that
encoders can be evaluated at different temporal granularities.
"""

import numpy as np
import pandas as pd

from torchgeo_bench.coordbench.benchmark import CoordBenchmark

RESOLUTION_LABELS = {
    "24h": "daily",
}

# Supported aggregation periods
TEMPORAL_AGGREGATION_METHODS = {
    "daily": ["1_week", "2_week", "4_week", "13_week"],
}


def _check_temporal_resolution(dataset: CoordBenchmark) -> str:
    """Infer the temporal resolution of a dataset from its timestamps.

    Args:
        dataset: Coordinate benchmark with ``posix_timestamp``

    Returns:
        The resolution label (e.g. ``"daily"``) from ``RESOLUTION_LABELS``.
    """
    df = pd.DataFrame({
        "lat": dataset.lat,
        "lon": dataset.lon,
        "timestamp": pd.to_datetime(dataset.posix_timestamp, unit="s"),
    })

    # Per-location gap between consecutive observations
    diffs = (
        df.sort_values("timestamp")
        .groupby(["lat", "lon"])["timestamp"]
        .diff()
        .dropna()
    )

    diffs_unique = diffs.unique()
    if len(diffs_unique) != 1:
        raise ValueError(
            f"{dataset.name!r} has inconsistent gaps between observations: "
            f"{sorted(diffs_unique)}"
        )

    offset = pd.tseries.frequencies.to_offset(pd.Timedelta(diffs_unique[0]))

    return RESOLUTION_LABELS[offset.freqstr]


def temporal_aggregation(dataset: CoordBenchmark, method: str) -> CoordBenchmark:
    """Aggregate a coordinate benchmark to a coarser temporal resolution.

    Args:
        dataset: Daily-resolution coordinate benchmark to aggregate.
        method: Target aggregation period, e.g. ``"1_week"``, ``"2_week"``, ``"4_week"``, or ``"13_week"``.

    Returns:
        A new `CoordBenchmark` with observations aggregated over the requested period.
    """
    return temporal_aggregation_all(dataset, [method])[0]


def temporal_aggregation_all(dataset: CoordBenchmark, methods: list[str]) -> list[CoordBenchmark]:
    """Aggregate a coordinate benchmark to several coarser temporal resolutions at once.

    Compute daily -> calendar-week once here and resuse

    Args:
        dataset: Daily-resolution coordinate benchmark to aggregate.
        methods: Target aggregation periods, e.g. ``["1_week", "13_week"]``.

    Returns:
        One new `CoordBenchmark` per requested method, in the same order.
    """
    if dataset.temporal_resolution is not None:
        assert dataset.temporal_resolution == "daily", (
            "Dataset must have daily temporal resolution for aggregation (right now)."
        )
    else:
        assert _check_temporal_resolution(dataset) == "daily", (
            "Dataset must have daily temporal resolution for aggregation (right now)."
        )

    for method in methods:
        assert method in TEMPORAL_AGGREGATION_METHODS["daily"], (
            f"Invalid aggregation method {method!r} for daily resolution."
        )

    weekly, agg_method = _collapse_to_weekly(dataset)
    return [_merge_weeks(dataset, weekly, agg_method, method) for method in methods]


def _collapse_to_weekly(dataset: CoordBenchmark) -> tuple[pd.DataFrame, dict]:
    """Collapse a daily-resolution dataset to one row per (location, week).

    Weeks are simple fixed 7-day bins counted from the dataset's earliest
    timestamp (day 0-6 -> week 0, day 7-13 -> week 1, ...), not calendar weeks.

    Returns:
        The per-(lat, lon, week_start) table and the column -> aggregator mapping
    """
    task_cols = list(dataset.tasks)
    df = pd.DataFrame({
        "lat": dataset.lat,
        "lon": dataset.lon,
        "timestamp": pd.to_datetime(dataset.posix_timestamp, unit="s"),
        **{c: dataset.tasks[c] for c in task_cols},
    })

    origin = df["timestamp"].min()
    week_number = (df["timestamp"] - origin).dt.days // 7
    df["week_start"] = origin + pd.to_timedelta(week_number * 7, unit="D")

    if dataset.task_type == "classification":
        # For classification, take the most frequent label in the period (mode)
        agg_method = {c: (lambda s: s.mode().iat[0]) for c in task_cols}
    else:
        # For regression, take the mean value in the period
        agg_method = {c: "mean" for c in task_cols}

    # Timestamps must be aggregated so we can collapse to weekly bins
    agg_method["timestamp"] = "mean"

    weekly = df.groupby(["lat", "lon", "week_start"]).agg(agg_method)

    return weekly.reset_index(), agg_method


def _merge_weeks(
    dataset: CoordBenchmark, weekly: pd.DataFrame, agg_method: dict, method: str
) -> CoordBenchmark:
    """Merge consecutive weeks into ``n``-week periods, ``n`` parsed from ``method``."""
    task_cols = list(dataset.tasks)
    n_weeks = int(method.split("_")[0])
    origin = weekly["week_start"].min()
    week_number = (weekly["week_start"] - origin).dt.days // 7

    period = (week_number // n_weeks).rename("period")

    aggregated = (
        weekly.groupby(["lat", "lon", period])
        .agg(agg_method)
        .reset_index()
    )

    # Drop the trailing period if it's cut short by the end of the data
    last_day = pd.to_datetime(dataset.posix_timestamp, unit="s").max()
    period_end = origin + pd.to_timedelta((aggregated["period"] + 1) * n_weeks * 7 - 1, unit="D")
    aggregated = aggregated[period_end <= last_day]

    return CoordBenchmark(
        name=f"{dataset.name}-{method}",
        lat=aggregated["lat"].to_numpy(np.float64),
        lon=aggregated["lon"].to_numpy(np.float64),
        tasks={c: aggregated[c].to_numpy() for c in task_cols},
        task_type=dataset.task_type,
        posix_timestamp=aggregated["timestamp"].astype("datetime64[s]").astype("int64").to_numpy(),
        test_mask=None,  # might want to add support for this at some point?
    )
