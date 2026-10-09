from typing import Callable

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

def median_and_iqr(series):
    median = np.median(series, axis=0)
    q1 = np.percentile(series, 25, axis=0)
    q3 = np.percentile(series, 75, axis=0)
    iqr = q3 - q1
    aggregated = np.concatenate([median, iqr],
                                axis=0)
    return aggregated

def covariance(series):
    return np.cov(series, rowvar=False)

def statistical(series):
    min = np.min(series, axis=0)
    max = np.max(series, axis=0)
    mean = np.mean(series, axis=0)
    std = np.std(series, axis=0)
    combined = np.concatenate([min, max, mean, std], axis=0)
    return combined

def mean(series):
    return np.mean(series, axis=0)

# Keeps non-aggregated embeddings in memory all at the same time
# Fast, but expensive.  Ex: 10M 256-dim float32 embeddings are ~10 GB
def _spatial_agg_with_all_emb(bench: CoordBenchmark,
                              method: Callable,
                              embeddings: np.ndarray
                              ):
    spatial_agg_map = {}

    non_emb = {}
    # embedding grouping and get one-off values
    for i in track(range(len(bench.lon)), f"{method} grouping by spatial key"):
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
    for spatial_key, timestamp in track(spatial_agg_map.keys(), f"{method} aggregating by spatial key"):
        key = (spatial_key, timestamp)
        group = np.array(spatial_agg_map[key])
        agg = method(group)
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

    updated_benchmark = CoordBenchmark(
        name=f"{bench.name}-spatial-{method}",
        lat=np.array(lat),
        lon=np.array(lon),
        posix_timestamp=np.array(timestamps),
        tasks=tasks,
        task_type=bench.task_type,
    )

    return updated_benchmark, np_emb


def generate_buffered_embeddings(encoder: LocationEncoder,
                                 lon: list,
                                 lat: list,
                                 timestamps: list,
                                 group_size: int,
                                 emb_agg_func: Callable):

    buffered_emb = encoder.encode(np.array(lon),
                                  np.array(lat),
                                  np.array(timestamps))

    print(f"Shape of buffered embed: {np.array(buffered_emb).shape}")

    embeddings = []
    # reshape embeddings to be 2D
    emb_per_group = np.reshape(np.array(buffered_emb), (-1, group_size, 256))
    print(f"Shape of grouped and buffered embed: {emb_per_group.shape}")
    for i in range(len(emb_per_group)):
        emb = emb_per_group[i]
        print(f"Shape of individual group of embeddings: {np.array(emb).shape}")
        embeddings.append(emb_agg_func(np.array(emb)))

    return embeddings



def spatial_aggregation(bench: CoordBenchmark,
                        method: str,
                        embeddings: np.ndarray | None,
                        encoder: LocationEncoder|None = None,
                        agg_at_same_time=False) ->  tuple[CoordBenchmark, np.ndarray]:



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

    if agg_at_same_time:
        # if temporally aggregated already (or pre-computed), use existing embeddings
        if embeddings is None:
            assert encoder is not None, "Need encoder if embeddings are not pre-computed"
            embeddings = encoder.encode(bench.lon, bench.lat, bench.posix_timestamp)

        embeddings = np.array(embeddings)

        return _spatial_agg_with_all_emb(bench, emb_agg_func, embeddings)

    df = pd.DataFrame({"lon": bench.lon,
                       "lat": bench.lat,
                       "posix_timestamp": bench.posix_timestamp,
                       bench.spatial_aggregation_key[0]: bench.spatial_aggregation_key[1]} |
                      bench.tasks)

    grouped = df.groupby([bench.spatial_aggregation_key[0], "posix_timestamp"])

    finalized_rows = []
    embeddings = []

    agg_dictionary = {"lon": "mean", "lat": "mean", "posix_timestamp": lambda x: x.iloc[0]}
    for task_name, _ in bench.tasks.items():
        agg_dictionary[task_name] = lambda x: x.iloc[0]

    buffer_size = 1000000 #~4 GB for 256-dim float32 embeddings; can expand upwards for increased speed
    current_buffer_count = 0

    buffer_lat = []
    buffer_lon = []
    buffer_timestamp = []
    group_size = 0
    j = 0

    # generates embeddings per aggregation group to avoid memory spike
    for i, group in track(grouped, f"{method} spatial aggregation"):
        group_size = group["lon"].size


        current_buffer_count += group_size
        j+= 1

        buffer_lat.extend(group["lat"])
        buffer_lon.extend(group["lon"])
        buffer_timestamp.extend(group["posix_timestamp"])
        finalized_rows.append(group.agg(agg_dictionary))

        # limit I/O to model with queries of ~1GB instead of per group
        if current_buffer_count >= buffer_size:

            emb = generate_buffered_embeddings(encoder=encoder, lon=buffer_lon, lat=buffer_lat,
                                               timestamps=buffer_timestamp, group_size=group_size,
                                               emb_agg_func=emb_agg_func)
            embeddings.extend(emb)

            # reset buffer
            buffer_lon = []
            buffer_lat = []
            buffer_timestamp = []
            current_buffer_count = 0

    if len(buffer_lon) != 0:
        # flush whatever remains in buffer (i.e. if dataset not evenly divisible by buffer size)
        remaining_emb = generate_buffered_embeddings(encoder=encoder, lon=buffer_lon, lat=buffer_lat,
                                                   timestamps=buffer_timestamp, group_size=group_size,
                                                   emb_agg_func=emb_agg_func)
        embeddings.extend(remaining_emb)

    finalized_df = pd.DataFrame(finalized_rows)
    finalized_embeddings = np.array(embeddings)

    updated_tasks = {}
    for task_name, _ in bench.tasks:
        updated_tasks[task_name] = finalized_df[task_name].to_numpy()

    updated_benchmark = CoordBenchmark(
        name=f"{bench.name}-spatial-{method}",
        lat=finalized_df["lat"].to_numpy(),
        lon=finalized_df["lon"].to_numpy(),
        posix_timestamp=finalized_df["posix_timestamp"].to_numpy(),
        tasks=updated_tasks,
        task_type=bench.task_type,
    )

    return updated_benchmark, finalized_embeddings