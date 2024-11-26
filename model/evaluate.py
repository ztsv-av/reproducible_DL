import torch

from data.data import get_mnist_dataloader
from model.model import MNISTFC
from utils.vars import DEVICE, MODEL_PATH
from utils.utils import set_global_seed

def evaluate_model(model=None, seed=None):
    """
    Evaluate the model and return accuracy.
    """
    # set the default seed
    set_global_seed()
    # get test data
    test_loader = get_mnist_dataloader(train=False)
    model = MNISTFC().to(DEVICE)
    # TODO: ensure that trained model file exists before running evaluate_model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total

    return accuracy
