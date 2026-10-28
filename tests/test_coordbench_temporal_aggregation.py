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
    n = len(days)
    values = np.tile(days.astype(float), 2)
    if task_type == "classification":
        values = (values >= 7).astype(int)
    return CoordBenchmark(
        name="toy",
        lat=np.repeat([10.0, 20.0], n),
        lon=np.repeat([30.0, 40.0], n),
        tasks={"y": values},
        task_type=task_type,
        temporal_resolution=TemporalResolution.DAILY,
        posix_timestamp=np.tile(days * DAY_SECONDS, 2),
    )


def test_aggregation_reduces_each_window_over_time():
    bench, embeddings = discretize_dataset_and_aggregate_embeddings(
        _daily_benchmark(), DayEncoder(), "weekly"
    )

    assert len(bench.lat) == 4
    # check that the embeddings are correct for each window
    # we can compute since the encoder is just the day number
    np.testing.assert_allclose(embeddings["mean"].numpy().ravel(), [3, 10, 3, 10]) # mean of days 0..6 is 3, mean of days 7..13 is 10
    np.testing.assert_allclose(embeddings["min"].numpy().ravel(), [0, 7, 0, 7])
    np.testing.assert_allclose(embeddings["max"].numpy().ravel(), [6, 13, 6, 13])


def test_aggregation_is_independent_of_batch_size():
    # to make sure I got the reshape and groupby logic correct
    # check results are the same for a small batch size
    reference = discretize_dataset_and_aggregate_embeddings(
        _daily_benchmark(), DayEncoder(), "weekly"
    )[1]["mean"]
    small = discretize_dataset_and_aggregate_embeddings(
        _daily_benchmark(), DayEncoder(), "weekly", batch_size=7
    )[1]["mean"]
    np.testing.assert_array_equal(small.numpy(), reference.numpy())


@pytest.mark.parametrize(
    ("task_type", "task"), [("regression", "y_mean"), ("classification", "y_mode")]
)
def test_expand_temporal_keeps_only_mean_targets(task_type, task):
    ((bench, features),) = _expand_temporal(
        [_daily_benchmark(task_type)], ["weekly"], [], ["mean"], encoder=DayEncoder()
    )

    assert list(bench.tasks) == [task]
    assert bench.temporal_resolution == TemporalResolution.WEEKLY
    assert features.shape == (4, 1) # 4 windows, 1 feature (the mean of the embeddings, or here, days, over time)


def test_expand_temporal_concatenates_statistics_under_a_distinct_name():
    results = list(
        _expand_temporal(
            [_daily_benchmark()], ["weekly"], [], ["mean", "mean_std_slope_tv"], encoder=DayEncoder()
        )
    )

    (mean_bench, mean_features), (concat_bench, concat_features) = results
    assert mean_bench.name != concat_bench.name
    assert concat_features.shape == (4, 4)  # mean, std, slope, total variation of a 1-D embedding
    # Window 0 is days 0..6: mean 3, std of 0..6, OLS slope on t in [-1, 1], total variation 6.
    expected = [3, np.arange(7).std(ddof=1), 3, 6]
    np.testing.assert_allclose(concat_features[0], expected)
    np.testing.assert_array_equal(concat_features[:, :1], mean_features)



def test_non_daily_sampling_without_a_spatial_key():
    """Most coarse benchmarks (CHELSA monthly, global population, nightlights) carry no key."""
    starts = np.array([pd.Timestamp("2017-01-01").timestamp()] * 2)
    bench = CoordBenchmark(
        name="yearly",
        lat=np.array([10.0, 20.0]),
        lon=np.array([30.0, 40.0]),
        tasks={"y": np.arange(2.0)},
        temporal_resolution=TemporalResolution.YEARLY,
        posix_timestamp=starts,
        end_timestamp=starts + 365 * DAY_SECONDS,
    )

    sampled = non_daily_temporal_sampling(bench, "monthly")

    assert sampled.posix_timestamp.shape == (2, 12)
    assert sampled.spatial_aggregation_key is None


def test_non_daily_sampling_drops_partial_weeks_across_every_field():
    """A short trailing window is dropped, so labels and the spatial key must drop with it."""
    starts = np.arange(4, dtype=float) * 7 * DAY_SECONDS
    ends = starts + 7 * DAY_SECONDS
    ends[-1] = starts[-1] + 3 * DAY_SECONDS  # final window covers only part of a week
    bench = CoordBenchmark(
        name="weekly",
        lat=np.arange(4.0),
        lon=np.arange(4.0),
        tasks={"y": np.arange(4.0)},
        temporal_resolution=TemporalResolution.WEEKLY,
        posix_timestamp=starts,
        end_timestamp=ends,
        spatial_aggregation_key=("county", pd.Series(["a", "b", "c", "d"])),
    )

    sampled = non_daily_temporal_sampling(bench, "daily")

    assert sampled.posix_timestamp.shape == (3, 7)
    np.testing.assert_array_equal(sampled.lat, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(sampled.tasks["y"], [0.0, 1.0, 2.0])
    # Spatial aggregation reads this by .iloc alongside plain arrays, so it stays a Series
    # whose positions line up with the kept rows.
    column, values = sampled.spatial_aggregation_key
    assert column == "county"
    assert values.iloc[2] == "c"
    assert len(values) == 3


def test_expand_temporal_samples_a_season_of_untimestamped_benchmarks():
    bench = CoordBenchmark(
        name="static", lat=np.zeros(3), lon=np.zeros(3), tasks={"y": np.arange(3.0)}, year=2017
    )
    ((sampled, features),) = _expand_temporal(
        [bench], ["weekly"], ["summer"], ["mean"], encoder=DayEncoder()
    )

    # Weekly grid over 2017 (Jan 1 + 7k days), restricted to June-August.
    grid = pd.date_range("2017-01-01", "2018-01-01", freq="7D", inclusive="left")
    summer = grid[grid.month.isin([6, 7, 8])]
    expected_mean = np.mean((summer - pd.Timestamp(0)) / pd.Timedelta(days=1))

    assert sampled.name == "static_summer_weekly_mean"
    np.testing.assert_allclose(features.ravel(), [expected_mean] * 3)
