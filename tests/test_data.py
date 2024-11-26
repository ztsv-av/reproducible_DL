import pytest

from data.data import compute_mean_std

def test_compute_mean_std():
    """
    Tests whether the computed mean and std of the MNIST dataset
    is the same as the ground truth values.
    """
    epsilon = 1e-4
    # compute mean and std
    mean, std = compute_mean_std()
    # expected values
    expected_mean = 0.1307
    expected_std = 0.3081
    # check that the computed mean and std are close to the expected values
    assert mean == pytest.approx(expected_mean, abs=epsilon), f"Computed mean {mean} is not close to expected {expected_mean}."
    assert std == pytest.approx(expected_std, abs=epsilon), f"Computed std {std} is not close to expected {expected_std}."
