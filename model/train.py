import torch
from torch.optim import AdamW, Optimizer
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss

import os

from data.data import get_mnist_dataloader
from model.model import MNISTFC
from model.utils import calculate_accuracy
from utils.vars import DEVICE, MODEL_PATH
from utils.utils import set_global_seed

def forward(data: Tensor, optimizer: Optimizer, model: Module) -> Tensor:
    """
    Forward path in model training.
    
    Parameters:
        - data (torch.Tensor): Input data batch.
        - optimizer (torch.optim.Optimizer): Optimizer for the model.
        - model (torch.nn.Module): PyTorch model instance.
    
    Returns:
        - outputs (torch.Tensor): Model's outputs after the forward pass.
    """
    optimizer.zero_grad()
    outputs = model(data)
    return outputs

def backward(outputs: Tensor, label: Tensor, criterion: _Loss, optimizer: Optimizer) -> None:
    """
    Backward path in model training.
    
    Parameters:
        - outputs (torch.Tensor): Model's outputs from the forward pass.
        - label (torch.Tensor): Ground truth labels corresponding to the input data.
        - criterion (orch.nn.modules.loss._Loss): Loss function to compute the training loss.
        - optimizer (torch.optim.Optimizer): Optimizer for the model.
    """
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()

def train_model(save: bool = True) -> float:
    """
    Train the MNISTFC model and return the trained model.
    The function does not accept any parameters to ensure reproducibility.
    Number of epochs, learning rate, loss function and optimizer are hardcoded to 
        2, 0.001, CrossEntropyLoss() and AdamW(), respectively.

    Parameters:
        save (bool): Whether to save the trained model or not.

    Returns:
        - accuracy (float): Cumilative accuracy on the training data.
    """
    # set the default seed
    set_global_seed()
    # get training data
    train_loader = get_mnist_dataloader(train=True)
    # define model and move it to default device
    model = MNISTFC().to(DEVICE)
    # define loss
    criterion = torch.nn.CrossEntropyLoss() # hardcoded loss function
    # define optimizer
    learning_rate = 0.001 # hardcoded learning rate
    optimizer = AdamW(model.parameters(), lr=learning_rate) # hardcoded optimizer

    # training loop
    epochs = 2 # hardcoded number of epochs
    correct = 0
    total = 0
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0  # accumulate loss
        correct = 0  # count correct predictions
        total = 0  # count total predictions
        for batch_idx, (data, label) in enumerate(train_loader):
            # move data to device
            data, label = data.to(DEVICE), label.to(DEVICE)
            # forward path
            outputs = forward(data, optimizer, model)
            # backward path
            backward(outputs, label, criterion, optimizer)
            # accuracy
            correct, total = calculate_accuracy(outputs, label, correct, total)
        # calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"    Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    accuracy = 100 * correct / total

    if save:
        # make sure directory where to save the model exists
        if not os.path.exists(os.path.dirname(MODEL_PATH)):
            os.makedirs(os.path.dirname(MODEL_PATH))
        # save trained model
        torch.save(model.state_dict(), MODEL_PATH)
    return accuracy

