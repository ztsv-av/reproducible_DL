import torch

import numpy as np
import random

from utils.vars import SEED

def set_global_seed(seed: int = SEED) -> None:
    """
    Sets the random seed for PyTorch, NumPy, and Python's random module to ensure reproducibility.
    
    Parameters:
        - seed (int): The seed value to set.
    """
    # set seed for Python's random module
    random.seed(seed)
    # set seed for NumPy
    np.random.seed(seed)
    # set seed for PyTorch
    torch.manual_seed(seed)
    # ensures PyTorch operations on the CPU are deterministic
    torch.use_deterministic_algorithms(True)