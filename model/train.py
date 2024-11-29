import torch
import torch.optim as optim

import os

from data.data import get_mnist_dataloader
from model.model import MNISTFC
from utils.vars import DEVICE, MODEL_PATH, EPOCHS, LR
from utils.utils import set_global_seed

def train_model(epochs=EPOCHS, lr=LR):
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
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print("\nStarting training...")
    # training loop
    correct = 0
    total = 0
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0  # accumulate loss
        correct = 0  # count correct predictions
        total = 0  # count total predictions
        for batch_idx, (data, label) in enumerate(train_loader):
            # forward
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            # backward
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            # metric
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        # calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"    Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    accuracy = 100 * correct / total

    # make sure directory where to save the model exists
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))
    # save trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Done!\nModel accuracy on the training set after training for {epochs} epochs: {accuracy:.2f}%")

