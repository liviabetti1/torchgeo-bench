"""Tests for models implemented in src/torchgeo_bench/coordbench/models.py.

Currently tests Climplicit and some basic functionality.
TODO: maybe write tests for the rest of the models?
"""

import numpy as np
import pytest

from torchgeo_bench.coordbench.models import (
    ClimplicitLocationEncoder,
    GTLocEncoder,
    SinCosLocationEncoder,
    TemporalSatCLIPEncoder,
    _date_to_posix_timestamp,
)


TEST_LAT_LONS = np.array(
    [
        [-90.0, -180.0],
        [37.77, -122.42],
        [0.0, 0.0],
        [48.85, 2.35],
        [90.0, 180.0],
    ]
)

LATITUDES = TEST_LAT_LONS[:, 0]
LONGITUDES = TEST_LAT_LONS[:, 1]

TIMESTAMPS = np.array(
    [_date_to_posix_timestamp(2020, m, 15) for m in (1, 7, 12, 6, 3)]
)


def test_timestamp_conversion() -> None:
    assert _date_to_posix_timestamp(1970, 1, 1) == 0.0
    assert _date_to_posix_timestamp(1970, 1, 2) == 86400.0


def test_sincos_ignores_timestamp() -> None:
    encoder = SinCosLocationEncoder()
    without_time = encoder.encode(LONGITUDES, LATITUDES)
    with_time = encoder.encode(LONGITUDES, LATITUDES, TIMESTAMPS)
    np.testing.assert_array_equal(without_time, with_time)


@pytest.mark.parametrize("encoder_class", [GTLocEncoder, TemporalSatCLIPEncoder])
def test_temporal_encoders_require_timestamp(encoder_class) -> None:
    encoder = encoder_class.__new__(encoder_class)  # Skip __init__; don't load weights.
    with pytest.raises(ValueError, match="posix_timestamp"):
        encoder._encode(LONGITUDES, LATITUDES, None)


@pytest.mark.slow
def test_climplicit_depends_on_month_but_not_year() -> None:
    encoder = ClimplicitLocationEncoder()
    n = len(LONGITUDES)

    january_2020 = np.full(n, _date_to_posix_timestamp(2020, 1, 15))
    january_2021 = np.full(n, _date_to_posix_timestamp(2021, 1, 15))
    july_2020 = np.full(n, _date_to_posix_timestamp(2020, 7, 15))

    january = encoder.encode(LONGITUDES, LATITUDES, january_2020)

    np.testing.assert_allclose(
        january,
        encoder.encode(LONGITUDES, LATITUDES, january_2021),
    )
    assert not np.allclose(
        january,
        encoder.encode(LONGITUDES, LATITUDES, july_2020),
    )