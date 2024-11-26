import torch

import pytest

from model.model import MNISTFC
from utils.vars import DEVICE

def test_model_input():
    """
    Tests the input shape of the MNIST model.
    In MNIST, the images are of shape (28x28), 
    so the input layer of the MNIST model should be (1, 1, 28, 28), i.e. (batch_size, channels, width, height).
    """
    # define random data with MNIST example shape (28x28) and move it to CPU
    expected_input = torch.randn(1, 1, 28, 28).to(DEVICE) # (batch_size, channels, width, height)
    # define mnist model and ensure model is on CPU
    model = MNISTFC().to(DEVICE)
    # check whether the MNIST model accepts expected_input
    try:
        # forward pass
        _ = model(expected_input)
    except Exception as e:
        pytest.fail(f"Model failed to process input of shape {expected_input.shape}: {e}.")

def test_model_output():
    """
    Tests the output shape of the MNIST model.
    In MNIST, the number of classes is 10, the length of the output vector should also be 10, probability for each class.
    """
    # define mnist model and ensure model is on CPU
    model = MNISTFC().to(DEVICE)
    # define random data with MNIST example shape (28x28) and move it to CPU
    sample_input = torch.randn(1, 1, 28, 28).to(DEVICE) # (batch_size, channels, width, height)
    # forward pass
    output = model(sample_input)
    # test
    expected_shape = (1, 10)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}."
