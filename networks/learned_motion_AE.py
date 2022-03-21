import torch
from torch import nn
import torch.nn.functional as F


class Compressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, y):
        n_batch, n_window = y.shape[:2]
        y = y.reshape((n_batch * n_window, -1))
        y = F.elu(self.linear0(y))
        y = F.elu(self.linear1(y))
        y = F.elu(self.linear2(y))
        y = self.linear3(y)
        return y.reshape((n_batch, n_window, -1))


class Decompressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        n_batch, n_window = x.shape[:2]
        x = x.reshape((n_batch * n_window, -1))
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x.reshape((n_batch, n_window, -1))


class Stepper(nn.Module):
    def __init__(self, param_size, hidden_size=512):
        super().__init__()

        self.linear0 = nn.Linear(param_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, param_size)

    def forward(self, x):
        n_batch, n_window = x.shape[:2]
        x = x.reshape((n_batch * n_window, -1))
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x.reshape((n_batch, n_window, -1))
