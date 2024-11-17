import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(12 * 7 * 7, 24)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(24, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 12 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x 