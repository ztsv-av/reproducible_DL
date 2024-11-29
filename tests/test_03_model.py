import torch

import pytest

from model.model import MNISTFC
from model.train import train_model
from model.evaluate import evaluate_model
from utils.vars import DEVICE, MODEL_PATH

def test_model_input():
    """
    Tests the input shape of the model used for the MNIST dataset.
    In MNIST, the images are of shape (28x28), 
        so the input layer of the MNIST model should be (1, 1, 28, 28), 
        i.e. (batch_size, channels, width, height).
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
    Tests the output shape of the model used for the MNIST dataset.
    In MNIST, the number of classes is 10, 
        the length of the output vector should also be 10, 
        probability for each class.
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

def test_train_accuracy():
    """
    Test for model.train.train_model() function.
    Checks whether train accuracy is the same as the expected accuracy.
    """
    expected_accuracy = 0.9647
    accuracy = train_model(save=False)
    assert accuracy == pytest.approx(expected_accuracy, abs=1e-4), f"Computed accuracy {accuracy} is not close to expected {expected_accuracy}."

def test_eval_accuracy():
    """
    Test for model.evaluate.evaluate_model() function.
    Checks whether train accuracy is the same as the expected accuracy.
    """
    expected_accuracy = 0.9692
    accuracy = evaluate_model()
    assert accuracy == pytest.approx(expected_accuracy, abs=1e-4), f"Computed accuracy {accuracy} is not close to expected {expected_accuracy}."

def test_load_model():
    """
    Tests that the trained model under path MODEL_PATH exists and that it has expected layers and weights.
    """
    # load test model
    test_model_path = "tests/test_data/mnist_test.pth"
    test_model = MNISTFC().to(DEVICE)
    test_model.load_state_dict(torch.load(test_model_path, map_location=DEVICE, weights_only=True))
    # 1. check if the path to the model exists
    try:
        trained_model = MNISTFC().to(DEVICE) # load trained model
        trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)) # load trained model weights
    except Exception as e:
        pytest.fail(f"Could not load model stored under {MODEL_PATH}: {e}.")
    # 2. compare weights between the trained model and the test model
    for (name1, param1), (name2, param2) in zip(test_model.named_parameters(), trained_model.named_parameters()):
        assert name1 == name2, f"Expected parameter name {name1}, got {name2}."
        assert torch.equal(param1, param2), f"Expected parameter value {param1}, got {param2}."
