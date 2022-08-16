import torch
import torch.nn as nn
import torch.jit as jit

class ConvNet(jit.ScriptModule):
    def __init__(
        self,
        input_dim,
        layers,
        output_dim=0,
        normalization=None,
        activation=nn.ReLU,
        dropout=None,
        global_pool=False
    ):
        super().__init__()

        layers = []
        for filters, kernel, stride in layers:
            layers.append(nn.Conv2d(input_dim, filters, kernel, stride, padding='same'))
            if normalization is not None:
                layers.append(normalization(filters))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout2d(dropout))
            input_dim = filters
        if global_pool:
            layers.append(nn.AdaptiveAvgPool2d((1,1)))
            layers.append(nn.Flatten())
        if output_dim > 0:
            if global_pool:
                layers.append(nn.Linear(input_dim, output_dim))
            else:
                layers.append(nn.Conv2d(input_dim, output_dim, 1, 1, padding='same'))
        self.layers = nn.Sequential(*layers)
        self.output_dim = output_dim or input_dim

    @jit.script_method
    def forward(self, x):
        return self.layers(x)
