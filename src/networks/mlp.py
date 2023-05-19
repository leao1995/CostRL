import torch
import torch.nn as nn
import torch.jit as jit

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim=0,
        normalization=None,
        activation=nn.ReLU,
        dropout=None
    ):
        super().__init__()

        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            if normalization is not None:
                layers.append(normalization(dim))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            input_dim = dim
        if output_dim > 0:
            layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self.output_dim = output_dim or hidden_dims[-1]

    def forward(self, x):
        return self.layers(x)