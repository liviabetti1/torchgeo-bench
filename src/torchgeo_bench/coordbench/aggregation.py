import numpy as np
import pandas as pd

from torchgeo_bench.coordbench.datasets import CoordBenchmark

RESOLUTION_LABELS = {
    "24h": "daily",
}


TEMPORAL_AGGREGATION_METHODS = {
    "daily": ["1_week", "2_week", "4_week", "13_week"],
}

def _check_temporal_resolution(dataset: CoordBenchmark) -> str:
    """Check the temporal resolution of the dataset."""
    df = pd.DataFrame({
        "lat": dataset.lat, "lon": dataset.lon,
        "timestamp": pd.to_datetime(dataset.posix_timestamp, unit="s")
    })

    # per-location gap between consecutive observations
    diffs = (
        df.sort_values("timestamp")
        .groupby(["lat", "lon"])["timestamp"]
        .diff()
        .dropna()
    )

    diffs_unique = diffs.unique()
    if len(diffs_unique) != 1:
        raise ValueError(f"{dataset.name!r} has inconsistent gaps between observations: {sorted(diffs_unique)}")

    offset = pd.tseries.frequencies.to_offset(pd.Timedelta(diffs_unique[0]))

    return RESOLUTION_LABELS[offset.freqstr]

def temporal_aggregation(dataset: CoordBenchmark, method: str):
    """Aggregate the dataset to the specified temporal resolution."""

    if dataset.temporal_resolution is not None:
        assert dataset.temporal_resolution == "daily", "Dataset must have daily temporal resolution for aggregation (right now)."
    else:
        assert _check_temporal_resolution(dataset) == "daily", "Dataset must have daily temporal resolution for aggregation (right now)."

    assert method in TEMPORAL_AGGREGATION_METHODS["daily"], f"Invalid aggregation method {method!r} for daily resolution."

    return _aggregate_daily_by_period(dataset, method)

def _aggregate_daily_by_period(dataset: CoordBenchmark, method: str) -> CoordBenchmark:
    """Aggregate the dataset by a specified period."""
    task_cols = list(dataset.tasks)
    df = pd.DataFrame({
        "lat": dataset.lat, "lon": dataset.lon,
        "timestamp": pd.to_datetime(dataset.posix_timestamp, unit="s"),
        **{c: dataset.tasks[c] for c in task_cols},
    })

    if dataset.task_type == "classification":
        # for classification, take the most frequent label in the period (mode)
        agg_method = {c: (lambda s: s.mode().iat[0]) for c in task_cols}
    else:
        # for regression, take the mean value in the period
        agg_method = {c: "mean" for c in task_cols}

    # need to aggregate timestampe so we can collapse to calendar weeks
    agg_method["timestamp"] = "mean"

    weekly = (
        df.groupby(["lat", "lon", pd.Grouper(key="timestamp", freq="W")])
        .agg(agg_method)
        .reset_index()
    )

    # merge consecutive weeks into n-week periods
    n_weeks = int(method.split("_")[0])
    week_start = weekly["timestamp"].min()
    week_number = ((weekly["timestamp"] - week_start).dt.days / 7).round().astype(int)
    period = (week_number // n_weeks).rename("period")

    aggregated = (
        weekly.groupby(["lat", "lon", period])
        .agg(agg_method)
        .reset_index()
    )

    return CoordBenchmark(
        name=f"{dataset.name}-{method}",
        lat=aggregated["lat"].to_numpy(np.float64),
        lon=aggregated["lon"].to_numpy(np.float64),
        tasks={c: aggregated[c].to_numpy() for c in task_cols},
        task_type=dataset.task_type,
        posix_timestamp=aggregated["timestamp"].astype("int64") // 10**9,
        test_mask=None, # might want to add support for this at some point?
    )
