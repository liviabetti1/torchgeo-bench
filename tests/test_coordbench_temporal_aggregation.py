"""Tests for src/torchgeo_bench/coordbench/temporal_aggregation.py and its use in run.py."""

import numpy as np
import pytest

from torchgeo_bench.coordbench import CoordBenchmark
from torchgeo_bench.coordbench.models import LocationEncoder
from torchgeo_bench.coordbench.run import _expand_temporal
from torchgeo_bench.coordbench.temporal_aggregation import (
    DAY_SECONDS,
    discretize_dataset_and_aggregate_embeddings,
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
        posix_timestamp=np.tile(days * DAY_SECONDS, 2),
    )


def test_aggregation_reduces_each_window_over_time():
    bench, embeddings = discretize_dataset_and_aggregate_embeddings(
        _daily_benchmark(), DayEncoder(), "weekly"
    )

    assert len(bench.lat) == 4
    np.testing.assert_allclose(embeddings["mean"].numpy().ravel(), [3, 10, 3, 10])
    np.testing.assert_allclose(embeddings["min"].numpy().ravel(), [0, 7, 0, 7])
    np.testing.assert_allclose(embeddings["max"].numpy().ravel(), [6, 13, 6, 13])


def test_aggregation_is_independent_of_batch_size():
    reference = discretize_dataset_and_aggregate_embeddings(
        _daily_benchmark(), DayEncoder(), "weekly"
    )[1]["mean"]
    small = discretize_dataset_and_aggregate_embeddings(
        _daily_benchmark(), DayEncoder(), "weekly", batch_size=7
    )[1]["mean"]
    np.testing.assert_array_equal(small.numpy(), reference.numpy())


def test_aggregation_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown"):
        discretize_dataset_and_aggregate_embeddings(
            _daily_benchmark(), DayEncoder(), "weekly", embedding_aggregation_method="bogus"
        )


@pytest.mark.parametrize(
    ("task_type", "task"), [("regression", "y_mean"), ("classification", "y_mode")]
)
def test_expand_temporal_keeps_only_mean_targets(task_type, task):
    ((bench, features),) = _expand_temporal(
        [_daily_benchmark(task_type)], ["weekly"], encoder=DayEncoder()
    )

    assert list(bench.tasks) == [task]
    assert bench.temporal_resolution == "weekly"
    assert features.shape == (4, 1)


def test_expand_temporal_passes_through_untimestamped_benchmarks():
    bench = CoordBenchmark(
        name="static", lat=np.zeros(3), lon=np.zeros(3), tasks={"y": np.arange(3.0)}
    )
    assert list(_expand_temporal([bench], ["weekly"], encoder=DayEncoder())) == [(bench, None)]
