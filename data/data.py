import torch
from torchvision import datasets, transforms

from utils.vars import DEVICE, MNIST_PATH

def compute_mean_std():
    """
    Computes the mean and standard deviation of the MNIST TRAINING dataset.

    Returns:
        - mean (float): Mean pixel value.
        - std (float): Standard deviation of pixel values.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST(
        root=MNIST_PATH, train=True, download=True, transform=transform)
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

def get_mnist_dataloader(batch_size=64, train=True):
    """
    Returns a DataLoader for the MNIST dataset, train or test.

    Parameters:
        - batch_size (int), default=64: Batch size for the DataLoader.
        - train (bool), default=True: Whether to load training or testing data.
    
    Returns:
        - loader (torch.utils.data.DataLoader): Dataloader for the MNIST dataset.
    """
    print("Downloading data...")
    # compute mean and std from TRAINING data and save them
    print("   Extracting mean, std...")
    mean, std = compute_mean_std() # always compute mean, std for the train part
    print("   Done!")
    # define the transform with the computed mean and std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    # define dataset
    dataset = datasets.MNIST(
        root=MNIST_PATH, train=train, download=True, transform=transform
    )
    # define dataloader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train  # shuffle only if training
    )
    print("Done!")
    return loader
