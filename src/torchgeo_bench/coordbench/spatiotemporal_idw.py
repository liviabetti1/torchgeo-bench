import numpy as np
import torch

KS = (1, 5, 20)
EARTH_R_KM = 6371.0

def pairwise_haversine_plus_time(s, s0, squared=False, return_separate=False):
    """Pairwise spatiotemporal distances.

    s: (N, 3), s0: (M, 3)
    Spatial distances in km; scaled_time in km-equivalent units.
    """
    lat0 = torch.deg2rad(s0[:, 0])[:, None]
    lon0 = torch.deg2rad(s0[:, 1])[:, None]
    lat = torch.deg2rad(s[:, 0])[None, :]
    lon = torch.deg2rad(s[:, 1])[None, :]

    dlat = lat - lat0
    dlon = lon - lon0

    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat0) * torch.cos(lat) * torch.sin(dlon / 2) ** 2
    )
    d_space = 2 * EARTH_R_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))
    d_time = s[None, :, 2] - s0[:, None, 2]  # (M, N) broadcasting

    if return_separate:
        return d_space.abs(), d_time.abs()

    d2 = d_space**2 + d_time**2
    return d2 if squared else torch.sqrt(d2)
        
def pairwise_haversine_plus_time_chunked(
    s, s0, k, batch_size=128, device="cuda", return_separate=False
):
    """Exact kNN search using chunked pairwise distances."""
    s = torch.as_tensor(s, dtype=torch.float32, device=device)
    s0 = torch.as_tensor(s0, dtype=torch.float32, device=device)

    all_dist, all_idx = [], []
    haversine_dist = []
    time_dist = []

    for start in range(0, len(s0), batch_size):
        q = s0[start:start + batch_size]

        if return_separate:
            d_space, d_time = pairwise_haversine_plus_time(s, q, squared=False, return_separate=True)
            vals_space, _ = torch.topk(d_space, k, largest=False, sorted=True)
            vals_time, _ = torch.topk(d_time, k, largest=False, sorted=True)

            haversine_dist.append(vals_space.cpu().numpy())
            time_dist.append(vals_time.cpu().numpy())
            continue

        dist = pairwise_haversine_plus_time(s, q, squared=False, return_separate=return_separate)
        vals, idx = torch.topk(dist, k, largest=False, sorted=True)

        all_dist.append(vals.cpu().numpy())
        all_idx.append(idx.cpu().numpy())

    if return_separate:
        # use to see if haversine and time are on comparable scales
        return np.concatenate(haversine_dist), np.concatenate(time_dist)

    return np.concatenate(all_dist), np.concatenate(all_idx)

def knn_predict_multi(
    lat_tr, lon_tr, time_tr, y_tr, lat_te, lon_te, time_te, ks, task_type: str, n_classes: int, c_param: float
) -> dict:
    """Predict with inverse great-circle distance weights for each k, sharing one query.
    Based on https://github.com/taylor-geospatial/mind/blob/main/scripts/paper/coord_interp_baseline.py"""

    # not sure what/if to standardize...
    # standardize time only
    # because haversine is in km
    # mu = np.mean(time_tr)
    # sigma = np.std(time_tr, ddof=1)

    # sqrt_c = np.sqrt(c_param)
    # t_tr = sqrt_c * (np.asarray(time_tr) - mu) / sigma
    # t_te = sqrt_c * (np.asarray(time_te) - mu) / sigma
    sqrt_c = np.sqrt(c_param)
    t_tr = sqrt_c * np.asarray(time_tr)
    t_te = sqrt_c * np.asarray(time_te)

    # Keep lat/lon in degrees; 
    # distance function converts to radians
    # check wrapping with test?
    S = np.stack([lat_tr, lon_tr, t_tr], axis=1)
    S0 = np.stack([lat_te, lon_te, t_te], axis=1)

    # k is the same as n_neigh in idwST
    # restrict to kmax
    # mind implementation does kmax = min(max(ks), len(lat_tr))
    kmax = max(ks)
    assert kmax <= len(lat_tr), "Invalid choice of k (number of neighbors)"

    # mind implementation parallelizes jobs
    # see https://github.com/taylor-geospatial/mind/blob/main/scripts/paper/coord_interp_baseline.py
    # this is a later problem...

    # Exact nearest neighbors on GPU
    distances, indices = pairwise_haversine_plus_time_chunked(
        S, S0, k=kmax
    )

    # A 10 m floor bounds weights for coincident coordinates.
    # not sure what this is, but mind does it?
    inv = 1.0 / np.maximum(distances, 0.01)
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