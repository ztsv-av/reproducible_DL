import torch

from utils.vars import SEED

def set_global_seed() -> None:
    """
    Sets the random seed for PyTorch to ensure reproducibility.
    """
    torch.manual_seed(SEED)
