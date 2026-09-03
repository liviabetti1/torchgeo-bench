# get t_satclip path from config somehow
# path should be /home/libe2152/projects/temporal-satclip/satclip

import os
import sys
from torch import nn

DEFAULT_POSIX_MIN_TIME = 1609487709.024  # Jan 1, 2021
DEFAULT_POSIX_MAX_TIME = 1767609639.024  # Dec 31, 2025

def load_t_satclip(
    ckpt_path: str,
    repo_path: str,                    # /home/libe2152/projects/temporal-satclip
    model_name: str = "tsatclip/doy",  # "tsatclip/linear" | "tsatclip/doy" | "tsatclip/toroidal"
    device: str = "cpu",
    posix_min_time: float = DEFAULT_POSIX_MIN_TIME,
    posix_max_time: float = DEFAULT_POSIX_MAX_TIME,
) -> nn.Module:
    for p in (repo_path, os.path.join(repo_path, "satclip")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from load_temporal_satclip_wrapper import TemporalSatCLIPWrapper

    model = TemporalSatCLIPWrapper(
        model_name=model_name,
        ckpt_path=ckpt_path,
        device=device,
        posix_min_time=posix_min_time,
        posix_max_time=posix_max_time,
    )
    model.eval()
    model.to(device)   # load_from_checkpoint ignores its own device arg — cast explicitly
    return model
