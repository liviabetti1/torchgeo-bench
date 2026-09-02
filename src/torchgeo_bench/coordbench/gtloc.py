"""GTLoc encoder for location and time features.

In the original paper, there is a separate location encoder and time encoder.
For our purposes, we concatenate these embeddings."""
import math

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from typing import Optional
from torch import Tensor


def posix_to_time_components(posix_timestamp: np.ndarray) -> np.ndarray:
    """Convert POSIX timestamps (seconds) to ``[month, day, hour, minute, second]`` columns."""
    ts = pd.to_datetime(np.asarray(posix_timestamp), unit="s")
    return np.stack(
        [ts.month.to_numpy(), ts.day.to_numpy(), ts.hour.to_numpy(), ts.minute.to_numpy(), ts.second.to_numpy()],
        axis=1,
    ).astype(np.float64)


# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336


def equal_earth_projection(L):
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180

# Constants
MINUTE_TO_HOUR  = 1 / 60
SECOND_TO_HOUR  = MINUTE_TO_HOUR / 60

DAY_TO_MONTH    = 1 / (365 / 12)
HOUR_TO_MONTH   = DAY_TO_MONTH / 24
MINUTE_TO_MONTH = MINUTE_TO_HOUR * HOUR_TO_MONTH
SECOND_TO_MONTH = SECOND_TO_HOUR * HOUR_TO_MONTH


def angular_time_representation(T):
    month, day, hour, minute, second = torch.chunk(T.float(), 5, dim=1)

    # Convert integer local datetime to month and day with decimal places
    month_d = (month - 1) + (day - 1) * DAY_TO_MONTH
    hour_d = hour + minute * MINUTE_TO_HOUR + second * SECOND_TO_HOUR

    # Transform month and day to angles between [-pi, +pi)
    theta = 2 * math.pi * month_d / 12 - math.pi
    phi = 2 * math.pi * hour_d / 24 - math.pi

    return torch.cat((theta, phi), dim=1)


def sample_b(sigma: float, size: tuple) -> Tensor:
    return torch.randn(size) * sigma

@torch.jit.script
def gaussian_encoding(
        v: Tensor,
        b: Tensor) -> Tensor:
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

class GaussianEncoding(nn.Module):
    def __init__(self, sigma: Optional[float] = None,
                 input_size: Optional[int] = None,
                 encoded_size: Optional[int] = None,
                 b: Optional[Tensor] = None):
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required.')

            b = sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        return gaussian_encoding(v, self.b)

class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma, embedding_dim=512):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=embedding_dim//2)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(embedding_dim, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, embedding_dim))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x

class TimeEncoderCapsule(nn.Module):
    def __init__(self, sigma, embedding_dim=512, dropout_prob=None):
        super(TimeEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=embedding_dim//2)
        self.sigma = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(embedding_dim, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob) if dropout_prob else nn.Identity(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob) if dropout_prob else nn.Identity(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob) if dropout_prob else nn.Identity())
        self.head = nn.Sequential(nn.Linear(1024, embedding_dim))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x


class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], embedding_dim=512):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        self.embedding_dim = embedding_dim

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s, embedding_dim=embedding_dim))

    def forward(self, location):
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], self.embedding_dim, device=location.device)

        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)
        
        return location_features

class TimeEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], embedding_dim=512, dropout_prob=None):
        super(TimeEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        self.embedding_dim = embedding_dim

        for i, s in enumerate(self.sigma):
            self.add_module('TimeEnc' + str(i), TimeEncoderCapsule(sigma=s, embedding_dim=embedding_dim, dropout_prob=dropout_prob))

    def forward(self, time):
        time = angular_time_representation(time)
        time_features = torch.zeros(time.shape[0], self.embedding_dim, device=time.device)

        for i in range(self.n):
            time_features += self._modules['TimeEnc' + str(i)](time)
        
        return time_features

class GTLoc(nn.Module):
    """Combined GTLoc encoder: location + time features, concatenated."""

    def __init__(
        self,
        loc_sigma=[2**0, 2**4, 2**8],
        time_sigma=[2**0, 2**4, 2**8],
        embedding_dim=512,
        time_dropout=None,
    ):
        super(GTLoc, self).__init__()
        self.embedding_dim = embedding_dim
        self.location_encoder = LocationEncoder(sigma=loc_sigma, embedding_dim=embedding_dim)
        self.time_encoder = TimeEncoder(
            sigma=time_sigma, embedding_dim=embedding_dim, dropout_prob=time_dropout
        )

    def forward(
        self, latlon: Tensor, posix_timestamp: Tensor | None = None
    ) -> Tensor:
        location_features = self.location_encoder(latlon)

        time = posix_to_time_components(posix_timestamp.cpu().numpy())
        time = torch.from_numpy(time).to(latlon.device)
        time_features = self.time_encoder(time)
        return torch.cat((location_features, time_features), dim=-1)


def load_gtloc(ckpt_path: str, device: str = "cpu") -> GTLoc:
    """Load a GTLoc checkpoint (``.pt``), inferring the location/time encoder shapes from the weights."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # The full training checkpoint also carries an image encoder and CLIP-style logit scales
    # We don't need those here

    # Count how many GTLocLocationEncoderCapsule modules are in the ckpt
    # GTLocLocationEncoderCapsule modules are named LocEnc0, LocEnc1, etc. in the state dict
    # and use different sigma values.
    # Each capsule has one head (so we count the number of heads)
    n_loc = sum(
        1 for k in state if k.startswith("location_encoder.LocEnc") and k.endswith(".head.0.weight")
    )

    # Same idea for the TimeEncoderCapsule modules, named TimeEnc0, TimeEnc1, etc.
    n_time = sum(
        1 for k in state if k.startswith("time_encoder.TimeEnc") and k.endswith(".head.0.weight")
    )

    # Index 0 of GTLocLocationEncoderCapsule is GaussianEncoding
    # Index 1 is nn.Linear(embedding_dim, 1024)
    embedding_dim = state["location_encoder.LocEnc0.capsule.1.weight"].shape[1]

    # Use sigma vals as placeholers, then load from ckpt into model
    model = GTLoc(loc_sigma=[1.0] * n_loc, time_sigma=[1.0] * n_time, embedding_dim=embedding_dim)
    encoder_state = {
        k: v for k, v in state.items()
        if k.startswith("location_encoder.") or k.startswith("time_encoder.")
    }
    model.load_state_dict(encoder_state)
    return model.to(device).eval()