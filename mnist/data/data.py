import torch
from torchvision import datasets, transforms

import os
import json

def compute_mnist_mean_std():
    """
    Computes the mean and standard deviation of the MNIST training dataset.

    Returns:
        - mean (float): Mean pixel value.
        - std (float): Standard deviation of pixel values.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=60000, shuffle=False)
    # load all images in one batch
    data_iter = iter(loader)
    images, _ = next(data_iter)
    # flatten images
    images = images.view(images.size(0), -1)
    # compute mean and std
    mean = images.mean().item()
    std = images.std().item()

    return mean, std

def save_mean_std(mean, std, filepath="mnist.json"):
    """
    Saves the mean and std to a JSON file.

    Parameters:
        - mean (float): Mean pixel value.
        - std (float): Standard deviation of pixel values.
        - filepath (str), default="mnist.json": Path to the json file storing mean and std of pixel values of the MNIST training set.
    """
    with open(filepath, "w") as f:
        json.dump({"mean": mean, "std": std}, f)

def load_mean_std(filepath="mnist.json"):
    """
    Loads the mean and std from a JSON file.

    Parameters:
        - filepath (str), default="mnist.json": Path to the json file storing mean and std of pixel values of the MNIST training set.

    Returns:
        - mean (float): Mean pixel value.
        - std (float): Standard deviation of pixel values.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["mean"], data["std"]

def get_mnist_dataloader(batch_size=64, train=True):
    """
    Returns a DataLoader for the MNIST dataset.

    Parameters:
        - batch_size (int), default=64: Batch size for the DataLoader.
        - train (bool), default=True: Whether to load training or testing data.
    
    Returns:
        - loader (torch.utils.data.DataLoader): Dataloader for the MNIST dataset.
    """
    data_parameters_file = "mnist.json"

    # check if mean and std have already been computed and stored
    if os.path.exists(data_parameters_file):
        mean, std = load_mean_std(data_parameters_file)
    else:
        # compute mean and std from TRAINING data and save them
        mean, std = compute_mnist_mean_std()
        save_mean_std(mean, std, data_parameters_file)
    # define the transform with the computed mean and std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    # define dataset
    dataset = datasets.MNIST(
        "./data", train=train, download=True, transform=transform
    )
    # define dataloader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train  # shuffle only if training
    )
    return loader
