import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

class Residual(nn.Module):  # @save
    def __init__(self, input_channels, num_channels, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=1, stride=1)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.bn3 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.bn3(self.conv3(X))

        Y += X

        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b1 = nn.Sequential(*resnet_block(870, 256, 16))
net = nn.Sequential(b1, nn.Conv1d(256, 16, kernel_size=1, stride=1), nn.Flatten(), nn.Linear(544, 34), nn.ReLU(), nn.Dropout(0.3), nn.Linear(34, 1))