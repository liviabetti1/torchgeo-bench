"""Tests for methods implemented in src/torchgeo_bench/coordbench/temporal_aggregation.py

TODO
"""

import numpy as np
import pandas as pd
import pytest

from torchgeo_bench.coordbench import CoordBenchmark
from torchgeo_bench.coordbench.models import LocationEncoder
from torchgeo_bench.coordbench.temporal_aggregation import (
    _check_temporal_resolution,
    method_to_windows,
    temporal_aggregation,
)
