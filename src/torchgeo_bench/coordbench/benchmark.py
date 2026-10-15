"""The :class:`CoordBenchmark` data model, shared by dataset loaders and transforms.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from torchgeo_bench.coordbench.config_enums import TemporalResolution


@dataclass
class CoordBenchmark:
    """A single coordinate -> label benchmark.

    Args:
        name: Unique benchmark identifier (e.g. ``"sustainbench-asset"``).
        lat: Latitudes, shape ``(N,)``.
        lon: Longitudes, shape ``(N,)``.
        tasks: Mapping of task/column name -> label array, each shape ``(N,)``.
        task_type: ``"regression"`` (R^2) or ``"classification"`` (accuracy).
        posix_timestamp: Optional per-point POSIX timestamp for time-conditioned encoders (Used to be year: Optional per-point year for year-conditioned encoders.)
        test_mask: Optional boolean held-out test mask (official split); when
            ``None`` the probe uses k-fold cross-validation.
        spatial_aggregation_key: Optional for polygon-based datasets, where many lon/lat pairs relate to the same entity.  Column name for grouping lon/lat pairs in aggregation
    """

    name: str
    lat: np.ndarray
    lon: np.ndarray
    tasks: dict[str, np.ndarray] = field(default_factory=dict)
    task_type: str = "regression"
    temporal_resolution: TemporalResolution | None = None
    year: int | None = None
    posix_timestamp: np.ndarray | None = None
    test_mask: np.ndarray | None = None
    spatial_aggregation_key: tuple[str, pd.Series] = None
    end_timestamp: np.ndarray | None = None

    def filter_by_latitude(self, min_lat: float) -> None:
        """Drop points below ``min_lat`` in-place."""
        mask = self.lat >= min_lat
        self.lat = self.lat[mask]
        self.lon = self.lon[mask]
        self.tasks = {k: v[mask] for k, v in self.tasks.items()}
        if self.posix_timestamp is not None:
            self.posix_timestamp = self.posix_timestamp[mask]
        if self.end_timestamp is not None:
            self.end_timestamp = self.end_timestamp[mask]
        if self.test_mask is not None:
            self.test_mask = self.test_mask[mask]
        if self.spatial_aggregation_key is not None:
            col, series = self.spatial_aggregation_key
            self.spatial_aggregation_key = (col, series[mask])
