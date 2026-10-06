import numpy as np
import pandas as pd
from multiprocessing import Pool

from torchgeo_bench.coordbench import CoordBenchmark, LocationEncoder

# Based on spatial pooling methods explored in https://arxiv.org/pdf/2603.02080
SPATIAL_AGGREGATION_METHODS = [
    "mean",
    "median_and_iqr",
    "covariance",
    "statistical"
]


def spatial_aggregation(bench: CoordBenchmark,
                        method: str,
                        embeddings: np.ndarray | None,
                        encoder: LocationEncoder|None = None) ->  tuple[CoordBenchmark, np.ndarray]:
    from IPython import embed; embed()

    #basic columns that will be present regardless of pre-processing steps
    df = pd.DataFrame({"lat": bench.lat,
                       "lon": bench.lon,
                       "posix_timestamp": bench.posix_timestamp,
                       bench.spatial_aggregation_key[0]: bench.spatial_aggregation_key[1], # tuple of column for spatial aggregation and column values; ex: county zipcode
                       "test_mask": bench.test_mask,
                       })

    for task, labels in bench.tasks.items():
        df[task] = labels # get the labels for this task

    # if temporally aggregated already (or pre-computed), use existing embeddings
    if embeddings is not None:
        df["emb"] = embeddings
    else:
        assert encoder is not None, "Need encoder if embeddings are not pre-computed"
        embeddings = encoder.encode(df["lon"], df["lat"], df["posix_timestamp"])
        df["emb"] = embeddings

    # Note that groupby will drop rows with key None
    grouped_by = df.groupby([bench.spatial_aggregation_key[0], "posix_timestamp"])

    # for parallel processing; needs to be nested for efficient access to global method
    def _aggregate_helper(spatial_key, timestamp: float, group_df: pd.DataFrame):
        lon = group_df["lon"].mean() # generate rough centroid for polygon if visual becomes necessary
        lat = group_df["lat"].mean()

        emb = group_df["emb"]
        if method == "mean":
            aggregated = np.mean(emb, axis=0)

        elif method == "median_and_iqr":
            median = np.median(emb, axis=0)
            q1 = np.percentile(emb, 25, axis=0)
            q3 = np.percentile(emb, 75, axis=0)
            iqr = q3 - q1
            aggregated = np.concatenate([median, iqr],
                                        axis=0)

        elif method == "covariance":
            aggregated = np.cov(emb, rowvar=True)

        elif method == "statistical":
            min = np.min(emb, axis=0)
            max = np.max(emb, axis=0)
            mean = np.mean(emb, axis=0)
            std = np.std(emb, axis=0)
            aggregated = np.concatenate([min, max, mean, std],
                                        axis=0)  # TODO: test how shape is handled downstream.  SHOULD be fine, but tbd, as this and median_and_iqr both change the feature shape
        else:
            raise NotImplementedError

        row = {"lon": lon, "lat": lat, "posix_timestamp": timestamp, bench.spatial_aggregation_key[0]: spatial_key, "emb": aggregated}

        # each of these will have the same value across every row of the group
        for task_name, _ in bench.tasks:
            row[task_name] = group_df[task_name].iloc[0]

        row['test_mask'] = group_df["test_mask"].iloc[0]

        return row


    with Pool() as P: # dynamically assigns one process per available core; could update to pass in num cores as a config
        finalized_df = pd.DataFrame(P.map(_aggregate_helper, grouped_by)) #TODO: needs testing.  Could be a simple loop, but worth exploring for efficiency's sake

    # need to grab single label per group for each task to ensure shape match
    grouped_tasks = {}
    for task, _ in bench.tasks:
        grouped_tasks[task] = finalized_df[task]

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

