import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only because some CUDA scatter kernels used by GCNConv have no deterministic variant.
    torch.use_deterministic_algorithms(True, warn_only=True)
