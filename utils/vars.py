import torch


# seed
SEED = 1
# device for pytorch
DEVICE = torch.device("cpu")
# mnist data path
MNIST_PATH = "data"
# trained model path
MODEL_PATH = "model/trained/mnist_model.pth"
# batch size for the train dataloader
BATCH_SIZE = 64
# number of epochs for training
EPOCHS = 2
# learning rate
LR = 0.01
