import torch
from torch import nn


class Compressor(nn.Module):
    def __init__(self, n_layers, n_units, activation_layer=nn.ELU, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_features=kwargs['input_shape'], out_features=n_units))
        self.layers.append(activation_layer())
        # hidden and output layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(in_features=n_units, out_features=n_units))
            self.layers.append(activation_layer())

    def forward(self, y):
        activation = y
        for layer in self.layers:
            activation = layer(activation)
        return activation


class Decompressor(nn.Module):
    def __init__(self, n_layers, n_units, activation_layer=nn.ReLU, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.x_size = kwargs['input_shape']
        # input layer
        self.layers.append(nn.Linear(in_features=(self.x_size+n_units), out_features=n_units))
        self.layers.append(activation_layer())
        # hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(nn.Linear(in_features=n_units, out_features=n_units))
            self.layers.append(activation_layer())
        # output layer
        self.layers.append(nn.Linear(in_features=n_units, out_features=kwargs['output_shape']))
        self.layers.append(activation_layer())

    def forward(self, x, z):
        v = torch.cat([x, z], 1) if z.dim() > 1 else torch.cat([x, z])
        return self.forward_input_vector(v)

    def forward_input_vector(self, v):
        activation = v
        for layer in self.layers:
            activation = layer(activation)
        return activation
