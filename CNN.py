import torch
from torch import nn
from torch.utils.data import DataLoader


def create_data_loader(train_data, batch_size):
    return DataLoader(train_data, batch_size=batch_size)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 10, kernel_size=5, stride=1)

        self.conv2 = nn.Sequential(
            nn.Conv1d(10, 20, kernel_size=32, stride=1),
            nn.MaxPool1d(kernel_size=1024, stride=1024)
        )

        self.flat = nn.Flatten()
        self.normalized = nn.BatchNorm1d(40)

        self.linear1 = nn.Linear(40, 4)
        self.active = nn.Tanh()
        self.linear2 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.normalized(x)
        x = self.linear1(x)
        x = self.active(x)
        logits = self.linear2(x)
        return logits
