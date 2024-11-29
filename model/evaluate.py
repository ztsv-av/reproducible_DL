import torch
from torch.nn import Module

from data.data import get_mnist_dataloader
from model.model import MNISTFC
from model.utils import calculate_accuracy
from utils.vars import DEVICE, MODEL_PATH
from utils.utils import set_global_seed

def load_model() -> Module:
    """
    Loads trained MNISTFC model stored under MODEL_PATH.

    Returns:
        model (torch.nn.Module): Loaded PyTorch model.
    """
    print("   Loading trained model...")
    model = MNISTFC().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    print("   Done!")

    return model

def evaluate_model() -> None:
    """
    Evaluate the model on the test data.

    Returns:
        - accuracy (float): Cumilative accuracy on the test data.
    """
    # set the default seed
    set_global_seed()
    # get test data
    test_loader = get_mnist_dataloader(train=False)

    # load trained model
    model = load_model()
    # evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_loader:
            # move data to device
            data, label = data.to(DEVICE), label.to(DEVICE)
            # get outputs
            outputs = model(data)
            # calculate accuracy
            correct, total = calculate_accuracy(outputs, label, correct, total)
    accuracy = 100 * correct / total
    return accuracy
