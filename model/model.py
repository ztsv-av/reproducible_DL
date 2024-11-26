import torch.nn as nn
import torch.nn.functional as F

class MNISTFC(nn.Module):
    """
    A simple neural network with one fully connected hidden layer and 1 output fully connected layer.
    """
    def __init__(self):
        super(MNISTFC, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
