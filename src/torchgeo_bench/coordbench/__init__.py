"""CoordBench: coordinate-only location-encoder evaluation for torchgeo-bench.

Loads the unified ``taylor-geospatial/coordbench`` benchmark suite (point
``(lon, lat)`` -> label) and probes a frozen coordinate encoder with KNN and a
ridge linear head under random or spatial-block cross-validation.

Public API
----------
.. autoclass:: LocationEncoder
.. autoclass:: SinCosLocationEncoder
.. autoclass:: CoordBenchmark
.. autofunction:: load_benchmarks
.. autofunction:: run_coordbench
"""

<<<<<<< HEAD
from torchgeo_bench.coordbench.datasets import (
    CoordBenchmark,
    list_benchmarks,
    list_families,
    load_benchmarks,
)
from torchgeo_bench.coordbench.models import (
    ClimplicitLocationEncoder,
    GeoCLIPLocationEncoder,
    LocationEncoder,
    MINDLocationEncoder,
    SatCLIPLocationEncoder,
    SinCosLocationEncoder,
    SINRLocationEncoder,
)
from torchgeo_bench.coordbench.probe import (
    knn_probe_score,
    linear_probe_score
)
from torchgeo_bench.coordbench.splits import spatial_fold_ids
from torchgeo_bench.coordbench.run import CoordResult, run_coordbench
=======
import lazy_loader as lazy
>>>>>>> origin/main

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
