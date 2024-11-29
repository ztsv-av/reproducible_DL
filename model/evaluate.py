import torch

from data.data import get_mnist_dataloader
from model.model import MNISTFC
from utils.vars import DEVICE, MODEL_PATH
from utils.utils import set_global_seed

def load_model():
    """
    Loads trained MNISTFC model stored under MODEL_PATH.
    """
    print("   Loading trained model...")
    model = MNISTFC().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    print("   Done!")
    return model

def evaluate_model():
    """
    Evaluate the model and return accuracy.
    """
    # set the default seed
    set_global_seed()
    # get test data
    test_loader = get_mnist_dataloader(train=False)

    print("\nStarting evaluation...")
    # load trained model
    model = load_model()
    # evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy = 100 * correct / total
    print(f"Done!\nModel accuracy on the validation set: {accuracy}%")
