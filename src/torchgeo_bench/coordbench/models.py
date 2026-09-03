"""Coordinate-only encoders for the CoordBench location-encoder track.

A :class:`LocationEncoder` maps points ``(lon, lat[, posix_timestamp])`` to a fixed-length
feature vector, one row per point; the probes and cross-validation live downstream.
Add a model by subclassing :class:`LocationEncoder`, implementing :meth:`_encode`,
and pointing a Hydra ``model`` config's ``_target_`` at it.

The trivial :class:`SinCosLocationEncoder` and the pretrained
:class:`MINDLocationEncoder` ship in the base install. The other pretrained
reference encoders (SatCLIP / GeoCLIP / Climplicit / SINR) are thin wrappers
over the ``rshf`` package and require the ``coordbench`` extra
(``pip install -e ".[coordbench]"``).
"""
import os
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class LocationEncoder(ABC):
    """Frozen coordinate encoder: ``(lon, lat[, posix_timestamp]) -> (N, D)`` features.

    Args:
        device: Torch device string for the forward pass.
        batch_size: Points per forward chunk.
    """

    #: Human-readable identifier recorded in result rows.
    name: str = "location_encoder"

    def __init__(self, device: str = "cpu", batch_size: int = 8192) -> None:
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.batch_size = int(batch_size)

    @abstractmethod
    def _encode(self, lon: np.ndarray, lat: np.ndarray, posix_timestamp: np.ndarray | None) -> np.ndarray:
        """Encode a single chunk of points; return ``(len(lon), D)`` float32.

        Args:
            lon: Longitudes, shape ``(B,)``.
            lat: Latitudes, shape ``(B,)``.
            posix_timestamp: Optional per-point posix_timestamp, shape ``(B,)`` (``None`` if the
                encoder is time-invariant).
        """
        raise NotImplementedError

    def encode(
        self, lon: np.ndarray, lat: np.ndarray, posix_timestamp: np.ndarray | None = None
    ) -> np.ndarray:
        """Encode all points, batching internally.

        Args:
            lon: Longitudes, shape ``(N,)``.
            lat: Latitudes, shape ``(N,)``.
            posix_timestamp: Optional per-point posix_timestamp, shape ``(N,)``.

        Returns:
            Feature matrix of shape ``(N, D)``, dtype float32.
        """
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        n = len(lon)
        out: list[np.ndarray] = []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            pts = None if posix_timestamp is None else np.asarray(posix_timestamp)[start:end]
            out.append(np.asarray(self._encode(lon[start:end], lat[start:end], pts), np.float32))
        return np.concatenate(out, axis=0) if out else np.empty((0, 0), np.float32)


class SinCosLocationEncoder(LocationEncoder):
    """Dependency-free baseline: ``[sin(lat), cos(lat), sin(lon), cos(lon)]``."""

    name = "sincos"

    def _encode(self, lon: np.ndarray, lat: np.ndarray, _posix_timestamp: np.ndarray | None) -> np.ndarray:
        lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
        return np.stack(
            [np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)], axis=1
        ).astype(np.float32)


class MINDLocationEncoder(LocationEncoder):
    """MIND location encoder (distilled from AlphaEarth/Climplicit/GeoCLIP/SINR).

    Loads a released checkpoint from the HuggingFace Hub. Two configs ship:
    ``mind`` (the 64-d Matryoshka deploy prefix of the pooled trunk) and
    ``mind_small`` (the distilled student's 128-d head output). ``feature`` picks
    the trunk (``pooled``) or the projected head (``head``); ``dim`` truncates the
    Matryoshka embedding.

    Args:
        repo: HuggingFace repo id holding the weights.
        filename: Checkpoint file within the repo (``.safetensors`` or ``.pt``).
        dim: Embedding width to keep (Matryoshka prefix).
        feature: ``"pooled"`` (trunk) or ``"head"`` (projected output).
        default_year: Year supplied to year-conditioned checkpoints when a
            benchmark carries no per-point year.
    """

    name = "mind"

    def __init__(
        self,
        repo: str = "isaaccorley/MIND",
        filename: str = "mind.safetensors",
        dim: int = 64,
        feature: str = "pooled",
        default_year: int = 2021,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from huggingface_hub import hf_hub_download

        from torchgeo_bench.coordbench.mind import load_mind

        self.model = load_mind(hf_hub_download(repo, filename), device=self.device)
        self.dim = int(dim)
        self.feature = feature
        self.default_year = default_year

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, posix_timestamp: np.ndarray | None) -> np.ndarray:
        latlon = torch.stack([torch.as_tensor(lat), torch.as_tensor(lon)], dim=1).float()
        pts = None
        if self.model.use_year:
            if posix_timestamp is None:
                default_ts = pd.Timestamp(year=self.default_year, month=1, day=1, tz="UTC").timestamp()
                posix_timestamp = np.full(len(lat), default_ts)
            pts = torch.as_tensor(np.broadcast_to(posix_timestamp, (len(lat),)), dtype=torch.float32)
            pts = pts.to(self.device)
        emb = self.model(latlon.to(self.device), pts, return_features=(self.feature == "pooled"))
        return emb.float().cpu().numpy()[:, : self.dim]


class GTLocEncoder(LocationEncoder):
    """GTLoc encoder: concatenated location + time features.

    Args:
        ckpt_path: Path to the checkpoint file (``.pt``). Falls back to the
            ``GTLOC_CKPT`` environment variable when not given.
        dim: Embedding dimension to keep from the concatenated
            ``[location_features, time_features]`` output. Defaults to the
            full concatenated width (``2 * embedding_dim``).
    """

    name = "gtloc"

    def __init__(
        self,
        ckpt_path: str | None = None,
        dim: int | None = None,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from torchgeo_bench.coordbench.gtloc import load_gtloc

        self.model = load_gtloc(ckpt_path, device=self.device)
        self.dim = int(dim) if dim is not None else 2 * self.model.embedding_dim

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, posix_timestamp: np.ndarray | None) -> np.ndarray:
        if posix_timestamp is None:
            raise ValueError("posix_timestamp is required for GTLocEncoder")
        latlon = torch.stack([torch.as_tensor(lat), torch.as_tensor(lon)], dim=1).float()
        posix_timestamp = torch.as_tensor(posix_timestamp).to(self.device)
        emb = self.model(latlon.to(self.device), posix_timestamp)
        return emb.float().cpu().numpy()[:, : self.dim]


class _RSHFEncoder(LocationEncoder):
    """Base for pretrained encoders loaded from the ``rshf`` package.

    Subclasses build ``self.model`` in :meth:`__init__` and set
    :attr:`coord_order` (``"lonlat"`` or ``"latlon"``). The default
    :meth:`_encode` stacks coordinates in that order and calls the model.
    """

    coord_order: str = "lonlat"
    dtype: torch.dtype = torch.float32

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, _posix_timestamp: np.ndarray | None) -> np.ndarray:
        first, second = (lon, lat) if self.coord_order == "lonlat" else (lat, lon)
        x = torch.stack([torch.as_tensor(first), torch.as_tensor(second)], dim=1).to(
            self.device, self.dtype
        )
        return self.model(x).float().cpu().numpy()

class ClimplicitLocationEncoder(_RSHFEncoder):
    """Climplicit climate-specialist encoder (CHELSA, ReSIREN) -> 256-d.

    Conditioned on the per-point month derived from ``posix_timestamp``
    (rather than the model's time-invariant 1024-d four-season embedding).
    Requires the ``coordbench`` extra (``rshf``).

    Changed from original coordbench implementation!
    """

    name = "climplicit"
    coord_order = "lonlat"
    dim = 256

    def __init__(
        self,
        repo: str = "Jobedo/climplicit",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from rshf.climplicit import Climplicit

        self.model = (
            Climplicit.from_pretrained(repo, config={"return_chelsa": False}).to(self.device).eval()
        )

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, posix_timestamp: np.ndarray | None) -> np.ndarray:
        if posix_timestamp is None:
            raise ValueError("posix_timestamp is required for ClimplicitLocationEncoder")
        lonlat = torch.stack([torch.as_tensor(lon), torch.as_tensor(lat)], dim=1).float()

        ts = pd.to_datetime(np.asarray(posix_timestamp), unit="s")
        month = torch.as_tensor(ts.month.to_numpy().copy(), dtype=torch.float32)

        emb = self.model(lonlat.to(self.device), month.to(self.device))
        return emb.float().cpu().numpy()[:, : self.dim]


class SINRLocationEncoder(_RSHFEncoder):
    """SINR location encoder (Cole et al. 2023) -> 256-d features.

    Requires the ``coordbench`` extra (``rshf``).
    """

    name = "sinr"
    coord_order = "lonlat"

    def __init__(
        self,
        repo: str = "MVRL/sinr-location-encoder-1000-cls",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from rshf.sinr import SINR, SINRConfig

        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        conf = SINRConfig(
            num_inputs=cfg["num_inputs"],
            num_filts=cfg["num_filts"],
            depth=cfg["depth"],
            num_classes=cfg["num_classes"],
        )
        self.model = SINR.from_pretrained(repo, config=conf).to(self.device).eval()

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, _posix_timestamp: np.ndarray | None) -> np.ndarray:
        from rshf.sinr import preprocess_locs

        x = torch.stack([torch.as_tensor(lon), torch.as_tensor(lat)], dim=1).float().to(self.device)
        return self.model(preprocess_locs(x), return_feats=True).float().cpu().numpy()


class GeoCLIPLocationEncoder(_RSHFEncoder):
    """GeoCLIP location encoder (Equal-Earth + RFF) -> 512-d. Input order (lat, lon).

    Requires the ``coordbench`` extra (``rshf``). Note the weights expect (lat, lon)
    despite the upstream docstring's "Lon/Lat" example.
    """

    name = "geoclip"
    coord_order = "latlon"

    def __init__(
        self,
        repo: str = "MVRL/geoclip-location-encoder",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from rshf.geoclip import GeoCLIP, GeoCLIPConfig
        from safetensors.torch import load_file

        # Build from config + weights explicitly: PyTorchModelHubMixin.from_pretrained
        # no longer reconstructs the GeoCLIPConfig under recent huggingface_hub.
        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        config = GeoCLIPConfig(
            sigma=cfg["sigma"],
            input_size=cfg["input_size"],
            encoded_size=cfg["encoded_size"],
            dim=cfg["dim"],
        )
        model = GeoCLIP(config)
        model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
        self.model = model.to(self.device).eval()


class SatCLIPLocationEncoder(_RSHFEncoder):
    """SatCLIP location encoder (spherical harmonics + SirenNet). Input order (lon, lat).

    Runs in float64 (the released weights are double precision). Requires the
    ``coordbench`` extra (``rshf``).
    """

    name = "satclip"
    coord_order = "lonlat"
    dtype = torch.float64

    def __init__(
        self,
        repo: str = "MVRL/satclip-loc-enc-vit16-l40",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from rshf.satclip import SatClip

        self.model = SatClip.from_pretrained(repo).double().to(self.device).eval()

class TemporalSatCLIPEncoder(LocationEncoder):
    """SpatioTemporal SatCLIP (t-SatCLIP) encoder.

    Loads ``TemporalSatCLIPWrapper`` from the upstream ``temporal-satclip`` checkout
    via :func:`torchgeo_bench.coordbench.t_satclip.load_t_satclip`.

    Args:
        ckpt_path: Path to the Lightning checkpoint (``.ckpt``).
        repo_path: Path to the ``temporal-satclip`` repo root (contains ``satclip/``).
        model_name: ``"tsatclip/linear"``, ``"tsatclip/doy"`` or ``"tsatclip/toroidal"``.
        dim: Embedding dimension to keep. Defaults to the model's full output width.
    """

    name = "t-satclip"

    def __init__(
        self,
        ckpt_path: str,
        repo_path: str,
        model_name: str = "tsatclip/doy",
        dim: int | None = None,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from torchgeo_bench.coordbench.t_satclip import load_t_satclip

        self.model = load_t_satclip(
            ckpt_path, repo_path=repo_path, model_name=model_name, device=self.device
        )
        self.dim = int(dim) if dim is not None else self.model.embedding_dim

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, posix_timestamp: np.ndarray | None) -> np.ndarray:
        if posix_timestamp is None:
            raise ValueError("posix_timestamp is required for TemporalSatCLIPEncoder")
        x = torch.stack(
            [torch.as_tensor(lat), torch.as_tensor(lon), torch.as_tensor(posix_timestamp)], dim=1
        ).double().to(self.device)
        emb = self.model(x)
        return emb.float().cpu().numpy()[:, : self.dim]
