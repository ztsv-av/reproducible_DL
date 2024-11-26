import torch
import torch.optim as optim

from data.data import get_mnist_dataloader
from model.model import MNISTFC
from utils.vars import DEVICE, MODEL_PATH
from utils.utils import set_global_seed

def train_model(epochs=2):
    """
    Train the MNISTFC model and return the trained model.

    Parameters:
        - epochs (int): Number of epochs to train the model.
    """
    # set the default seed
    set_global_seed()
    # get training data
    train_loader = get_mnist_dataloader(train=True)
    # define model and move it to default device
    model = MNISTFC().to(DEVICE)
    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    return model
