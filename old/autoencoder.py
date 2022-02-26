import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, hidden_nodes=32, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs['input_shape'], out_features=hidden_nodes
        )
        self.encoder_output_layer = nn.Linear(
            in_features=hidden_nodes, out_features=hidden_nodes
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=hidden_nodes, out_features=hidden_nodes
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden_nodes, out_features=kwargs['input_shape']
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
