"""Setup needed for MPS to work."""

import torch
from torch.utils.data._utils.collate import default_collate_fn_map


def setup_mps():
    torch.set_default_dtype(torch.float32)

    def default_collate_fn(batch, **kwargs):
        return torch.tensor(batch, dtype=torch.float32)

    default_collate_fn_map[float] = default_collate_fn
