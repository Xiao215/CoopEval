"""Reproducibility helpers shared by repository scripts."""

from __future__ import annotations

import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible script runs."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
