import numpy as np
import pandas as pd
from rich.progress import track

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
    combined = np.concatenate([min, max, mean, std], axis=0)
    return combined

def mean(series):
    return np.mean(series, axis=0)


def spatial_aggregation(bench: CoordBenchmark,
                        method: str,
                        embeddings: np.ndarray | None,
                        encoder: LocationEncoder|None = None) ->  tuple[CoordBenchmark, np.ndarray]:

    # if temporally aggregated already (or pre-computed), use existing embeddings
    if embeddings is None:
        assert encoder is not None, "Need encoder if embeddings are not pre-computed"
        embeddings = encoder.encode(bench.lon, bench.lat, bench.posix_timestamp)

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

    embeddings = np.array(embeddings)

    spatial_agg_map = {}

    non_emb = {}
    # embedding grouping and get one-off values
    for i in track(range(len(bench.lon)), "grouping by spatial key"):
        spatial_key = bench.spatial_aggregation_key[1].iloc[i]
        timestamp = bench.posix_timestamp[i]
        embedding = embeddings[i]

        key = (spatial_key, timestamp)

        if key not in spatial_agg_map.keys():
            spatial_agg_map[key] = [embedding]

            # get first value of non-embedding columns: tasks, lon, lat, timestamp, spatial key
            row_of_non_emb = {}
            for task_name, values in bench.tasks.items():
                row_of_non_emb[task_name] = values[i]
            row_of_non_emb["lon"] = bench.lon[i]
            row_of_non_emb["lat"] = bench.lat[i]
            non_emb[key] = row_of_non_emb

        else:
            spatial_agg_map[key].append(embedding)

    emb = []
    lon = []
    lat = []
    timestamps = []
    tasks = {}
    # aggregation
    for spatial_key, timestamp in track(spatial_agg_map.keys(), "aggregating by spatial key"):
        key = (spatial_key, timestamp)
        group = np.array(spatial_agg_map[key])
        agg = emb_agg_func(group)
        emb.append(agg)

        # non-embedding aggregation to maintain proper ordering
        lon.append(non_emb[key]["lon"])
        lat.append(non_emb[key]["lat"])
        timestamps.append(timestamp)
        for task_name, values in bench.tasks.items():
            task_value = non_emb[key][task_name]
            if task_name in tasks:
                tasks[task_name].append(task_value)
            else:
                tasks[task_name] = [task_value]

    for task, values in bench.tasks.items():
        tasks[task] = np.array(tasks[task])

    np_emb = np.array(emb)
    print("shape of final", np_emb.shape)
    print(tasks)


    updated_benchmark = CoordBenchmark(
        name=f"{bench.name}-spatial-{method}",
        lat=np.array(lat),
        lon=np.array(lon),
        posix_timestamp=np.array(timestamps),
        tasks= tasks,
        task_type=bench.task_type,
    )

    return updated_benchmark, np_emb

