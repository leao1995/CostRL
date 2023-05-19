import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F
import torch.distributions as D

from nflows.nn import nets
from nflows import utils
from nflows import transforms

from src.networks.mlp import MLP
from src.networks.xformer import SAB

class RNNLatent(nn.Module):
    def __init__(
        self,
        input_dim,
        rnn_hidden_dim,
        num_rnn_layers,
        hidden_dims,
        latent_dim
    ):
        super().__init__()

        self.rnn = nn.GRU(input_dim, rnn_hidden_dim, num_rnn_layers, batch_first=True, bidirectional=False)
        self.latent = MLP(rnn_hidden_dim, hidden_dims, latent_dim*2)
        self.latent_dim = latent_dim

    def forward(self, x):
        assert x.ndim == 3
        hidden, _ = self.rnn(x)
        latent = self.latent(hidden)
        m, s = latent.chunk(2, dim=-1)
        s = F.softplus(s)
        dist = D.Independent(D.Normal(loc=m, scale=s), 1) # [B,T]
        return dist

class AttnLatent(nn.Module):
    def __init__(
        self,
        input_dim,
        attn_hidden_dims,
        hidden_dims,
        latent_dim,
        num_heads=4,
        ln=False
    ):
        super().__init__()

        layers = []
        for dim in attn_hidden_dims:
            layers.append(SAB(input_dim, dim, num_heads, ln=ln, causal=True))
            input_dim = dim
        layers.append(MLP(input_dim, hidden_dims, latent_dim*2))
        self.layers = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x):
        assert x.ndim == 3
        latent = self.layers(x)
        m, s = latent.chunk(2, dim=-1)
        s = F.softplus(s)
        dist = D.Independent(D.Normal(loc=m, scale=s), 1) # [B,T]
        return dist

def build_prior_trans(dim, hidden_dims):
    trans = []
    for i, hdim in enumerate(hidden_dims):
        trans.append(transforms.LULinear(dim))
        trans.append(transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(dim, even=(i%2==0)),
            transform_net_create_fn=lambda in_dim, out_dim: nets.ResidualNet(
                    in_features=in_dim,
                    out_features=out_dim,
                    hidden_features=hdim
                ),
            num_bins=8,
            tails='linear',
            tail_bound=3.0,
            apply_unconditional_transform=True
        ))
    trans.append(transforms.LULinear(dim))
    trans = transforms.CompositeTransform(trans)
    return trans

def build_posterior_trans(dim, hidden_dims):
    trans = []
    for i, hdim in enumerate(hidden_dims):
        trans.append(transforms.LULinear(dim))
        trans.append(transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=dim,
            hidden_features=hdim,
            num_bins=8,
            tails='linear',
            tail_bound=3.0
        ))
    trans.append(transforms.LULinear(dim))
    trans = transforms.CompositeTransform(trans)
    return transforms.InverseTransform(trans)

class CatDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_classes_per_dim
    ):
        super().__init__()
        self.num_classes_per_dim = num_classes_per_dim
        output_dim = sum(num_classes_per_dim)
        self.layer = MLP(input_dim, hidden_dims, output_dim)

    def forward(self, x):
        logits = self.layer(x)
        logits = logits.split(self.num_classes_per_dim, dim=-1)
        dist = [D.Categorical(logits=logit) for logit in logits]
        return dist
