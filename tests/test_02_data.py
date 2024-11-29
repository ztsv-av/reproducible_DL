import torch

import pytest

from data.data import compute_mean_std, get_mnist_dataloader

def test_compute_mean_std():
    """
    Test for data.compute_mean_std function.
    Checks whether the computed mean and std of the MNIST dataset
        is the same as the ground truth values.
    """
    # compute mean and std
    mean, std = compute_mean_std()
    # expected values
    expected_mean = 0.1307
    expected_std = 0.3081
    # check that the computed mean and std are close to the expected values
    assert mean == pytest.approx(expected_mean, abs=1e-4), f"Computed mean {mean} is not close to expected {expected_mean}."
    assert std == pytest.approx(expected_std, abs=1e-4), f"Computed std {std} is not close to expected {expected_std}."

def test_get_mnist_dataloader():
    """
    Test for data.get_mnist_dataloader() function. 
    Checks the consistency for:
        1. Dataset size.
        2. Data shape and target label type.
        3. Target label distribution.
    """
    # generate dataloader and test it
    retured_dataloader = get_mnist_dataloader(train=True)

    # 1. check dataset size
    dataset = retured_dataloader.dataset
    num_samples = len(dataset)
    expected_samples = 60000  # MNIST training set size
    assert num_samples == expected_samples, f"Expected number of samples is {expected_samples}, got {num_samples}."

    # 2. check shape and type of examples
    first_image, first_label = dataset[0]
    assert first_image.shape == torch.Size([1, 28, 28]), f"Expected image shape [1, 28, 28], got {first_image.shape}."
    assert isinstance(first_label, int), f"Expected label to be of type int, got {type(first_label)}."

    # 3. check target class distribution
    expected_class_counts = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
    class_counts = [0] * 10  # MNIST has 10 classes
    for _, label in dataset:
        class_counts[label] += 1
    assert class_counts == expected_class_counts, f"Class count mismatch: expected {expected_class_counts}, got {class_counts}."
