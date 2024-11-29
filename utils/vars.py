import torch


# seed for reproducibility
SEED = 1
# define the device globally
DEVICE = torch.device("cpu")
# mnist data path
MNIST_PATH = "data"
# trained mnist model path
MODEL_PATH = "model/trained/mnist_model.pth"
# number of epochs for training
EPOCHS = 2
# learning rate
LR = 0.01
