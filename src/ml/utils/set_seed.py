import random

import numpy as np
import torch


def set_seed(seed: int = 0):
    """Sets a seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)  # type: ignore
