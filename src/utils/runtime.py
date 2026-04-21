import codecs
import random

import numpy as np
import torch
from numpy.core.multiarray import _reconstruct


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_torch_checkpoint_safe_globals():
    """Allow trusted local Trainer checkpoints to restore NumPy RNG state."""
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    add_safe_globals(
        [
            _reconstruct,
            np.ndarray,
            np.dtype,
            np.uint32,
            type(np.dtype(np.uint32)),
            codecs.encode,
        ]
    )

