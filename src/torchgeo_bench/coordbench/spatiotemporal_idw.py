import numpy as np
import faiss

KS = (1, 5, 20)
EARTH_R_KM = 6371.0

SECONDS_IN_A_HOUR = 3600
SECONDS_IN_A_DAY = 86400
SECONDS_IN_A_YEAR = 31556952 # Gregorian year 365.2425 days

def lat_lon_to_xyz(lat, lon):
    """Convert lat/lon to 3D Cartesian coordinates."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = EARTH_R_KM * np.cos(lat_rad) * np.cos(lon_rad)
    y = EARTH_R_KM * np.cos(lat_rad) * np.sin(lon_rad)
    z = EARTH_R_KM * np.sin(lat_rad)
    return np.stack([x, y, z], axis=1)

# the following code is from VeeVee
# at https://github.com/arizonat/satclip/blob/temporal/satclip/load_temporal_satclip_wrapper.py

def _get_year_from_posix(posix_time):
    return np.floor(np.asarray(posix_time) / SECONDS_IN_A_YEAR).astype(int) + 1970

def time_to_normalized_year(posix_time, posix_min_time, posix_max_time):
    year = _get_year_from_posix(posix_time)
    min_year = _get_year_from_posix(posix_min_time)
    max_year = _get_year_from_posix(posix_max_time)
    return (year - min_year) / (max_year - min_year)  # Normalize to [0, 1]

def time_to_normalized_day_of_year(posix_time, posix_min_time, posix_max_time):
    day_of_year = ((posix_time % SECONDS_IN_A_YEAR) / SECONDS_IN_A_DAY).astype(int) + 1
    return (day_of_year - 1) / 364.0  # Normalize to [0, 1]

def time_to_normalized_time_of_year(posix_time, posix_min_time, posix_max_time):
    time_of_year = ((posix_time % SECONDS_IN_A_YEAR) / SECONDS_IN_A_DAY).astype(int) + 1
    return (time_of_year - 1) / 364.0  # Normalize to [0, 1]

def time_normalized(posix_time, posix_min_time, posix_max_time):
    normalized_time = (posix_time - posix_min_time) / (posix_max_time - posix_min_time)
    return normalized_time

def time_of_year_and_year(posix_time, posix_min_time, posix_max_time):
    return np.column_stack([
        time_to_normalized_time_of_year(posix_time, posix_min_time, posix_max_time),
        time_to_normalized_year(posix_time, posix_min_time, posix_max_time),
    ])

def find_knn(lat_tr, lon_tr, time_tr, lat_te, lon_te, time_te, kmax, c_param, device="cuda", time_fn=None):
    # time_fn: optional (posix_time, posix_min_time, posix_max_time) -> (n,) or (n, d) array

    # not sure what/if to standardize...for now no standardization
    # maybe later: standardize time only

    cartesian_tr = lat_lon_to_xyz(lat_tr, lon_tr)
    cartesian_te = lat_lon_to_xyz(lat_te, lon_te)

    if time_tr is None or time_te is None:
        print("No time provided, using spatial only")
        t_tr = np.zeros_like(lat_tr)
        t_te = np.zeros_like(lat_te)
    else:
        time_tr = np.asarray(time_tr, dtype=np.float64)
        time_te = np.asarray(time_te, dtype=np.float64)
        if time_fn is not None:
            tmin, tmax = time_tr.min(), time_tr.max()
            time_tr, time_te = time_fn(time_tr, tmin, tmax), time_fn(time_te, tmin, tmax)
        t0 = time_tr.min()
        sqrt_c = np.sqrt(c_param)
        # subtract t0 to avoid float32 overflow for large timestamps
        t_tr = sqrt_c * (time_tr - t0)
        t_te = sqrt_c * (time_te - t0)

    # Keep lat/lon in degrees; 
    # distance function converts to radians
    # check wrapping with test?
    S = np.column_stack([cartesian_tr, t_tr]).astype(np.float32)
    S0 = np.column_stack([cartesian_te, t_te]).astype(np.float32)

    # k is the same as n_neigh in idwST
    # restrict to kmax
    # mind implementation does kmax = min(max(ks), len(lat_tr))
    assert kmax <= len(lat_tr), "Invalid choice of k (number of neighbors)"

    # mind implementation parallelizes jobs
    # see https://github.com/taylor-geospatial/mind/blob/main/scripts/paper/coord_interp_baseline.py
    # this is a later problem...

    # Exact nearest neighbors on GPU
    index = faiss.IndexFlatL2(S.shape[1])

    if device == "cuda":
        res = faiss.StandardGpuResources()
        # assumes gpu id is 0
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(S)
    distances_sq, indices = index.search(S0, kmax)

    return np.sqrt(distances_sq), indices

def knn_predict(distances, indices, ks, y_tr, lat_te, lon_te, task_type: str, n_classes: int) -> dict:
    """ """
    kmax = max(ks)

    # A 10 m floor bounds weights for coincident coordinates.
    # not sure what this is, but mind does it?
    inv = 1.0 / np.maximum(distances, 0.01)
    # indices are from the training set, so we can index into y_tr to get the neighbor labels
    neigh = np.asarray(y_tr)[indices]

    out = {}
    for k in ks:
        k_eff = min(k, kmax)
        w = inv[:, :k_eff]
        w = w / w.sum(axis=1, keepdims=True)
        nb = neigh[:, :k_eff]

        if task_type == "regression":
            out[k] = (w * nb).sum(axis=1)
            continue

        votes = np.zeros((len(lat_te), n_classes))
        for j in range(k_eff):
            np.add.at(votes, (np.arange(len(lat_te)), nb[:, j]), w[:, j])
        out[k] = votes.argmax(axis=1)
    return out

def find_knn_and_predict(
    lat_tr, lon_tr, time_tr, y_tr, lat_te, lon_te, time_te, ks, task_type: str, n_classes: int, c_param: float, device: str = "cuda", time_fn=None
) -> dict:
    """Predict with inverse great-circle distance weights for each k, sharing one query.
    Based on https://github.com/taylor-geospatial/mind/blob/main/scripts/paper/coord_interp_baseline.py"""

    kmax = max(ks)
    distances, indices = find_knn(lat_tr, lon_tr, time_tr, lat_te, lon_te, time_te, kmax, c_param=c_param, device=device, time_fn=time_fn)

    return knn_predict(distances, indices, ks, y_tr, lat_te, lon_te, task_type=task_type, n_classes=n_classes)