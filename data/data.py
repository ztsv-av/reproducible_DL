import torch
from torchvision import datasets, transforms

from utils.vars import DEVICE, BATCH_SIZE

def compute_mean_std():
    """
    Computes the mean and standard deviation of the MNIST training dataset.

    Returns:
        - mean (float): Mean pixel value.
        - std (float): Standard deviation of pixel values.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST("./data", train=True, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=60000, shuffle=False)
    # load all images in one batch
    data_iter = iter(loader)
    images, _ = next(data_iter)
    # flatten images
    images = images.view(images.size(0), -1).to(DEVICE)
    # compute mean and std
    mean = images.mean().item()
    std = images.std().item()

    return mean, std

def get_mnist_dataloader(batch_size=BATCH_SIZE, train=True):
    """
    Returns a DataLoader for the MNIST dataset.

    Parameters:
        - batch_size (int), default=64: Batch size for the DataLoader.
        - train (bool), default=True: Whether to load training or testing data.
    
    Returns:
        - loader (torch.utils.data.DataLoader): Dataloader for the MNIST dataset.
    """
    # compute mean and std from TRAINING data and save them
    mean, std = compute_mean_std()
    # define the transform with the computed mean and std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    # define dataset
    dataset = datasets.MNIST(
        "./data", train=train, download=False, transform=transform
    )
    # define dataloader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train  # shuffle only if training
    )
    return loader
