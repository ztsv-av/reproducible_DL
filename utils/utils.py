import torch

import numpy as np
import random

from utils.vars import SEED

def set_global_seed() -> None:
    """
    Sets the random seed for PyTorch, NumPy, and Python's random module to ensure reproducibility.
    """
    # set seed for Python's random module
    random.seed(SEED)
    # set seed for NumPy
    np.random.seed(SEED)
    # set seed for PyTorch
    torch.manual_seed(SEED)
    # ensures PyTorch operations on the CPU are deterministic
    torch.use_deterministic_algorithms(True)