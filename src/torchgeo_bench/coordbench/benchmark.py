"""The :class:`CoordBenchmark` data model, shared by dataset loaders and transforms.

Split out from ``datasets.py`` so that modules which transform an existing
benchmark (e.g. ``aggregation.py``, ``splits.py``) can import the type without
importing the dataset loaders (and vice versa), avoiding a circular import.
"""

from dataclasses import dataclass, field

import numpy as np


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
    """

    name: str
    lat: np.ndarray
    lon: np.ndarray
    tasks: dict[str, np.ndarray] = field(default_factory=dict)
    task_type: str = "regression"
    temporal_resolution: str | None = None
    year: int | None = None
    posix_timestamp: np.ndarray | None = None
    test_mask: np.ndarray | None = None
