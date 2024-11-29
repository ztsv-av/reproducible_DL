import torch
from torch import Tensor

def calculate_accuracy(output: Tensor, label: Tensor, correct: int, total: int) -> tuple[int, int]:
    """
    Calculates the correct and total counts for accuracy calculation.
    Used during model training and evaluation.

    Parameters:
        - output (Tensor): Model's output logits.
        - label (Tensor): Ground truth labels for the batch.
        - correct (int): Current count of correct predictions.
        - total (int): Current count of total samples.

    Returns:
        - correct (int): Updated count of correct predictions.
        - total (int): Updated count of total samples.
    """
    _, predicted = torch.max(output.data, 1)
    total += label.size(0)
    correct += (predicted == label).sum().item()
    return correct, total
