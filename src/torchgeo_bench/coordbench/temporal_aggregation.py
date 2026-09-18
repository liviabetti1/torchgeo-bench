"""Daily -> weekly -> n-week aggregation of labels, timestamps, or embeddings.

Labels/timestamps aggregate by mean (regression) or mode (classification).

If an `encoder` is passed, embeddings are computed per-day and mean-pooled over the same windows.
"""

import os

import numpy as np
import pandas as pd
from rich.progress import track

from torchgeo_bench.coordbench.benchmark import CoordBenchmark
from torchgeo_bench.coordbench.models import LocationEncoder

RESOLUTION_LABELS = {"24h": "daily"}

DEFAULT_AGGREGATION_YEAR = 2021

# Representative (month-day) dates used to synthesize a timestamp for datasets that have none.
# Chosen by me, but possibly change??
SEASON_REPRESENTATIVE_DATES = {
    "winter": "01-15",
    "spring": "04-15",
    "summer": "07-15",
    "fall": "10-15",
}

TEMPORAL_AGGREGATION_METHODS = {
    "daily": ["1_week",
              "2_week",
              "4_week",
              "13_week",
              "annual",
              "concat_four_seasons"],
    "none": ["annual",
             "summer",
             "default_date_winter",
             "default_date_summer",
             "concat_four_seasons"],
}


def _check_temporal_resolution(dataset: CoordBenchmark) -> str:
    """Infer the temporal resolution of a dataset from its timestamps."""
    df = pd.DataFrame({
        "lat": dataset.lat,
        "lon": dataset.lon,
        "timestamp": pd.to_datetime(dataset.posix_timestamp, unit="s"),
    })
    diffs = df.sort_values("timestamp").groupby(["lat", "lon"])["timestamp"].diff().dropna()
    diffs_unique = diffs.unique()
    if len(diffs_unique) != 1:
        raise ValueError(f"{dataset.name!r} has inconsistent gaps between observations: {sorted(diffs_unique)}")
    offset = pd.tseries.frequencies.to_offset(pd.Timedelta(diffs_unique[0]))
    return RESOLUTION_LABELS[offset.freqstr]

def method_to_windows(method: str, year: int = 2021) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a calendar year into consecutive (start, end) windows of the method's length (e.g. "2_week").

    "annual" returns a single window spanning the full year; "summer" returns a single window
    spanning June-August (JJA).
    """
    if method in ("annual", "concat_four_seasons"):
        return [(pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year}-12-31", tz="UTC"))]
    if method == "summer":
        return [(pd.Timestamp(f"{year}-06-01", tz="UTC"), pd.Timestamp(f"{year}-08-31", tz="UTC"))]
    if method in ("default_date_winter", "default_date_summer"):
        season = method.removeprefix("default_date_")
        date = pd.Timestamp(f"{year}-{SEASON_REPRESENTATIVE_DATES[season]}", tz="UTC")
        return [(date, date)]

    n_days = int(method.split("_")[0]) * 7
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", tz="UTC")

    # Drop the last window if it's too short
    num_windows = len(days) // n_days
    return [(days[i * n_days], days[i * n_days + n_days - 1]) for i in range(num_windows)]


def _static_aggregation(
    dataset: CoordBenchmark,
    method: str,
    encoder: LocationEncoder | None,
    year: int = DEFAULT_AGGREGATION_YEAR,
) -> tuple[CoordBenchmark, np.ndarray | None]:
    """Aggregate a dataset with no per-point timestamp.
    """
    assert encoder is not None, (
        f"{method!r} requires an encoder to synthesize embeddings for {dataset.name!r} "
        "(it has no timestamp of its own)."
    )
    lon, lat = dataset.lon, dataset.lat

    if method == "concat_four_seasons":
        embs = [
            encoder.encode(lon, lat, np.full(len(lon), pd.Timestamp(f"{year}-{date}", tz="UTC").timestamp()))
            for date in track(SEASON_REPRESENTATIVE_DATES.values(), description="concat_four_seasons")
        ] # another option here is to get averaged over months...
        emb = np.concatenate(embs, axis=1)
    else:
        start, end = method_to_windows(method, year)[0]
        days = pd.date_range(start, end, freq="D")
        emb = np.mean(
            [encoder.encode(lon, lat, np.full(len(lon), d.timestamp())) for d in track(days, description=method)],
            axis=0,
        )

    bench = CoordBenchmark(
        name=f"{dataset.name}-windowed",
        lat=lat,
        lon=lon,
        tasks=dict(dataset.tasks),
        task_type=dataset.task_type,
        posix_timestamp=None,
    )
    return bench, emb.astype(np.float32)

def _nonstatic_aggregation(
    dataset: CoordBenchmark,
    method: str,
    encoder: LocationEncoder | None,
) -> tuple[CoordBenchmark, np.ndarray | None]:
    """Aggregate a dataset with a per-point (daily) timestamp into windows.
    """
    task_cols = list(dataset.tasks)
    df = pd.DataFrame({
        "lat": dataset.lat,
        "lon": dataset.lon,
        "timestamp": pd.to_datetime(dataset.posix_timestamp, unit="s", utc=True),
        **dataset.tasks,
    })

    label_agg = (
        {c: (lambda s: s.mode().iat[0]) for c in task_cols}
        if dataset.task_type == "classification"
        else {c: "mean" for c in task_cols}
    )

    all_labels, all_embs, all_ts = [], [], []
    list_of_windows = method_to_windows(method, dataset.year)
    for start, end in track(list_of_windows, description="temporal_aggregation_by_windows"):
        g = df[df["timestamp"].between(start, end)].groupby(["lat", "lon"])
        labels = g.agg(label_agg).reset_index()
        all_labels.append(labels)

        if method == "concat_four_seasons":
            assert encoder is not None, (
                f"{method!r} requires an encoder to synthesize embeddings for {dataset.name!r}."
            )
            lon_l, lat_l = labels["lon"].to_numpy(), labels["lat"].to_numpy()
            if encoder.name == "climplicit":
                # Climplicit's native no-month call already concatenates months 3/6/9/12.
                all_embs.append(encoder.encode(lon_l, lat_l, None))
            else:
                embs = [
                    encoder.encode(
                        lon_l,
                        lat_l,
                        np.full(len(labels), pd.Timestamp(f"{dataset.year}-{date}", tz="UTC").timestamp()),
                    )
                    for date in track(SEASON_REPRESENTATIVE_DATES.values(), description="concat_four_seasons")
                ]
                all_embs.append(np.concatenate(embs, axis=1))
        elif encoder is not None:
            days = pd.date_range(start, end, freq="D")
            all_embs.append(np.mean(
                [
                    encoder.encode(labels["lon"].to_numpy(), labels["lat"].to_numpy(), np.full(len(labels), d.timestamp()))
                    for d in days
                ],
                axis=0,
            ))
        else:
            all_ts.append((g["timestamp"].mean().astype("int64")).to_numpy())

    labels = pd.concat(all_labels, ignore_index=True)
    emb = np.concatenate(all_embs) if encoder is not None else None
    posix_timestamp = np.concatenate(all_ts) if encoder is None else None

    bench = CoordBenchmark(
        name=f"{dataset.name}-windowed", # maybe make name more specific?
        lat=labels["lat"].to_numpy(),
        lon=labels["lon"].to_numpy(),
        tasks={c: labels[c].to_numpy() for c in task_cols},
        task_type=dataset.task_type,
        posix_timestamp=posix_timestamp,
    )
    return bench, emb


def temporal_aggregation(
    dataset: CoordBenchmark,
    method: str,
    encoder: LocationEncoder | None = None,
) -> tuple[CoordBenchmark, np.ndarray | None]:
    """Average each window's labels (mean/mode); mean-pool per-day embeddings if `encoder` is given.
    """
    resolution = dataset.temporal_resolution or (
        "none" if dataset.posix_timestamp is None else _check_temporal_resolution(dataset)
    )
    if resolution == "none":
        return _static_aggregation(dataset, method, encoder)

    assert resolution == "daily", "Dataset must have daily temporal resolution for aggregation for now."
    return _nonstatic_aggregation(dataset, method, encoder)