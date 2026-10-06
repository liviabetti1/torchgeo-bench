import numpy as np
import pandas as pd

from torchgeo_bench.coordbench import CoordBenchmark, LocationEncoder

# Based on spatial pooling methods explored in https://arxiv.org/pdf/2603.02080
SPATIAL_AGGREGATION_METHODS = [
    "mean",
    "median_and_iqr",
    "covariance",
    "statistical"
]

method_parallelized = None

def median_and_iqr(series):
    median = np.median(series, axis=0)
    q1 = np.percentile(series, 25, axis=0)
    q3 = np.percentile(series, 75, axis=0)
    iqr = q3 - q1
    aggregated = np.concatenate([median, iqr],
                                axis=0)
    return aggregated

def covariance(series):
    return np.cov(series, rowvar=True)

def statistical(series):
    min = np.min(series, axis=0)
    max = np.max(series, axis=0)
    mean = np.mean(series, axis=0)
    std = np.std(series, axis=0)
    return np.concatenate([min, max, mean, std],
                                axis=0)

def mean(series):
    return np.mean(series, axis=0)


def spatial_aggregation(bench: CoordBenchmark,
                        method: str,
                        embeddings: np.ndarray | None,
                        encoder: LocationEncoder|None = None) ->  tuple[CoordBenchmark, np.ndarray]:

    #basic columns that will be present regardless of pre-processing steps
    df = pd.DataFrame({"lat": bench.lat,
                       "lon": bench.lon,
                       "posix_timestamp": bench.posix_timestamp,
                       bench.spatial_aggregation_key[0]: bench.spatial_aggregation_key[1], # tuple of column for spatial aggregation and column values; ex: county zipcode
                       "test_mask": bench.test_mask,
                       })

    for task in bench.tasks:
        df[task[0]] = task[1] # task is tuple of col_name, column

    # if temporally aggregated already (or pre-computed), use existing embeddings
    if embeddings is not None:
        df["emb"] = embeddings
    else:
        assert encoder is not None, "Need encoder if embeddings are not pre-computed"
        embeddings = encoder.encode(df["lon"], df["lat"], df["posix_timestamp"])
        df["emb"] = embeddings

    if method == "mean":
        emb_agg_func = mean
    elif method == "median_and_iqr":
        emb_agg_func = median_and_iqr
    elif method == "covariance":
        emb_agg_func = covariance
    elif method == "statistical":
        emb_agg_func = statistical
    else:
        raise NotImplementedError

    agg_dictionary = {"lon": "mean", "lat": "mean", "posix_timestamp": "first", "emb": emb_agg_func}
    for task, _ in bench.tasks:
        agg_dictionary[task[0]] = "first"

    finalized_df = df.groupby([bench.spatial_aggregation_key[0], "posix_timestamp"]).agg(agg_dictionary)

    updated_benchmark = CoordBenchmark(
        name=f"{bench.name}-spatial-{method}",
        lat=finalized_df["lat"],
        lon=finalized_df["lon"],
        posix_timestamp=finalized_df["posix_timestamp"],
        tasks=grouped_tasks,
        task_type=bench.task_type,
        test_mask=bench.test_mask,
    )

    return updated_benchmark, finalized_df["emb"].astype(np.float32)

