"""Tests for src/torchgeo_bench/coordbench/temporal_aggregation.py and its use in run.py."""

import numpy as np
import pandas as pd
import pytest

from torchgeo_bench.coordbench import CoordBenchmark
from torchgeo_bench.coordbench.config_enums import TemporalResolution
from torchgeo_bench.coordbench.models import LocationEncoder
from torchgeo_bench.coordbench.run import _expand_temporal
from torchgeo_bench.coordbench.temporal_aggregation import (
    DAY_SECONDS,
    discretize_dataset_and_aggregate_embeddings,
    non_daily_temporal_sampling,
)


class DayEncoder(LocationEncoder):
    """Encodes each point as its timestamp in days, so window statistics are known exactly."""

    def _encode(self, lon, lat, posix_timestamp):
        return (np.asarray(posix_timestamp, dtype=np.float64) / DAY_SECONDS)[:, None]


def _daily_benchmark(task_type: str = "regression") -> CoordBenchmark:
    """Two locations observed daily for days 0..14: two full weekly windows plus one dropped day."""
    days = np.arange(15)
    values = np.tile(days.astype(float), 2)
    return CoordBenchmark(
        name="toy",
        lat=np.repeat([10.0, 20.0], 15),
        lon=np.repeat([30.0, 40.0], 15),
        tasks={"y": values if task_type == "regression" else (values >= 7).astype(int)},
        task_type=task_type,
        temporal_resolution=TemporalResolution.DAILY,
        posix_timestamp=np.tile(days * DAY_SECONDS, 2),
    )


@pytest.mark.parametrize("batch_size", [None, 7])
def test_aggregation_reduces_each_window_over_time(batch_size: int | None) -> None:
    kwargs = {} if batch_size is None else {"batch_size": batch_size}
    bench, embeddings = discretize_dataset_and_aggregate_embeddings(
        _daily_benchmark(), DayEncoder(), "weekly", **kwargs
    )
    assert len(bench.lat) == 4
    np.testing.assert_allclose(embeddings["mean"].numpy().ravel(), [3, 10, 3, 10])
    np.testing.assert_allclose(embeddings["min"].numpy().ravel(), [0, 7, 0, 7])
    np.testing.assert_allclose(embeddings["max"].numpy().ravel(), [6, 13, 6, 13])


@pytest.mark.parametrize(
    ("task_type", "task"), [("regression", "y_mean"), ("classification", "y_mode")]
)
def test_expand_temporal_keeps_only_mean_targets(task_type: str, task: str) -> None:
    ((bench, features),) = _expand_temporal(
        [_daily_benchmark(task_type)], ["weekly"], [], ["mean"], encoder=DayEncoder()
    )
    assert list(bench.tasks) == [task]
    assert bench.temporal_resolution == TemporalResolution.WEEKLY
    assert features.shape == (4, 1)


def test_expand_temporal_concatenates_statistics_under_a_distinct_name() -> None:
    (mean_bench, mean_features), (concat_bench, concat_features) = _expand_temporal(
        [_daily_benchmark()], ["weekly"], [], ["mean", "mean_std_slope_tv"], encoder=DayEncoder()
    )
    assert mean_bench.name != concat_bench.name
    # Window 0 is days 0..6: mean 3, std of 0..6, OLS slope on t in [-1, 1], total variation 6.
    assert concat_features.shape == (4, 4)
    np.testing.assert_allclose(concat_features[0], [3, np.arange(7).std(ddof=1), 3, 6])
    np.testing.assert_array_equal(concat_features[:, :1], mean_features)


def test_expand_temporal_samples_a_static_benchmark_within_its_season() -> None:
    bench = CoordBenchmark(
        name="static",
        lat=np.array([10.0, 20.0]),
        lon=np.array([30.0, 40.0]),
        tasks={"y": np.arange(2.0)},
        year=2017,
    )
    ((sampled, features),) = _expand_temporal(
        [bench], ["monthly"], ["summer"], ["mean"], encoder=DayEncoder()
    )
    june_1, aug_1 = (pd.Timestamp(f"2017-{m}-01").timestamp() / DAY_SECONDS for m in (6, 8))
    np.testing.assert_allclose(features[:, 0], [(june_1 + june_1 + 30 + aug_1) / 3] * 2)
    np.testing.assert_array_equal(sampled.tasks["y"], [0.0, 1.0])


@pytest.mark.parametrize("key", [None, ("county", pd.Series(["a", "b", "c", "d"]))])
def test_non_daily_sampling_drops_partial_weeks_across_every_field(key: tuple | None) -> None:
    """A short trailing window is dropped, so labels and any spatial key must drop with it."""
    starts = np.arange(4, dtype=float) * 7 * DAY_SECONDS
    ends = starts + 7 * DAY_SECONDS
    ends[-1] = starts[-1] + 3 * DAY_SECONDS
    bench = CoordBenchmark(
        name="weekly", lat=np.arange(4.0), lon=np.arange(4.0), tasks={"y": np.arange(4.0)},
        temporal_resolution=TemporalResolution.WEEKLY, posix_timestamp=starts, end_timestamp=ends,
        spatial_aggregation_key=key,
    )  # fmt: skip
    sampled = non_daily_temporal_sampling(bench, "daily")
    assert sampled.posix_timestamp.shape == (3, 7)
    np.testing.assert_array_equal(sampled.lat, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(sampled.tasks["y"], [0.0, 1.0, 2.0])
    if key is None:
        assert sampled.spatial_aggregation_key is None
    else:
        # Spatial aggregation reads this by .iloc alongside plain arrays.
        column, values = sampled.spatial_aggregation_key
        assert (column, values.tolist()) == ("county", ["a", "b", "c"])
