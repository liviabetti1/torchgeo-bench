import numpy as np
import pandas as pd
from rich.progress import track

from torchgeo_bench.coordbench.models import LocationEncoder


def year_posix_timestamps(year: int = 2021, freq: str = "D", months: tuple[int, ...] | None = None) -> np.ndarray:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=freq, tz="UTC") + pd.Timedelta(hours=12)
    if months is not None:
        days = days[days.month.isin(months)]
    return (days.astype("int64") // 10**9).to_numpy()


def yearly_embeddings(
    encoder: LocationEncoder,
    latlon: np.ndarray,
    year: int,
    method: str = "mean",
    freq: str = "D",
    months: tuple[int, ...] | None = None,
) -> np.ndarray:
    # only supports method mean at the moment
    assert method == "mean"
    latlon = np.asarray(latlon)
    lat, lon = latlon[:, 0], latlon[:, 1]
    ts = year_posix_timestamps(year, freq, months)

    # loop one timestamp at a time (rather than tiling everything at once) so peak
    # memory is O(n_locations, D) instead of O(n_locations * n_timestamps, D)
    total = None
    for t in track(ts, description="yearly_embeddings"):
        emb = encoder.encode(lon, lat, np.full(len(lat), t))
        total = emb if total is None else total + emb
    return (total / len(ts)).astype(np.float32)


def summer_embeddings(
    encoder: LocationEncoder, latlon: np.ndarray, year: int, method: str = "mean", freq: str = "D"
) -> np.ndarray:
    return yearly_embeddings(encoder, latlon, year, method, freq, months=(6, 7, 8, 9))
