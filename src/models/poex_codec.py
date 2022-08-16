import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F
import torch.distributions as D

from nflows.nn import nets
from nflows import utils
from nflows import transforms

from src.networks.xformer import ISAB, PMA

class LatentEncoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        peq_hidden_dims, 
        latent_dim,
        num_heads=8,
        num_inds=16,
        ln=False
    ):
        super().__init__()
        
        peq_layers = []
        for dim in peq_hidden_dims:
            peq_layers.append(ISAB(input_dim, dim, num_heads, num_inds, ln))
            input_dim = dim
        self.peq_layers = nn.Sequential(*peq_layers)
        self.peq_embed_dim = input_dim

        pin_layers = []
        pin_layers.append(PMA(input_dim, num_heads, 1, ln))
        pin_layers.append(nn.Flatten())
        pin_layers.append(nn.Linear(input_dim, latent_dim*2))
        pin_layers.append(nn.ReLU(inplace=True))
        pin_layers.append(nn.Linear(latent_dim*2, latent_dim*2))
        self.pin_layers = nn.Sequential(*pin_layers)
        self.latent_dim = latent_dim

    def forward(self, x):
        assert x.ndim == 3, 'assume inputs are of size [B,T,d]'
        peq_embed = self.peq_layers(x)
        ms = self.pin_layers(peq_embed)
        m, s = ms.split(self.latent_dim, dim=1)
        s = F.softplus(s)
        dist = D.Independent(D.Normal(loc=m, scale=s), 1)
        
        return {
            'dist': dist,
            'peq_embed': peq_embed
        }

class PeqEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        num_heads=8,
        num_inds=16,
        ln=False
    ):
        super().__init__()
        
        layers = []
        for dim in hidden_dims:
            layers.append(ISAB(input_dim, dim, num_heads, num_inds, ln))
            input_dim = dim
        layers.append(ISAB(input_dim, output_dim, num_heads, num_inds, ln))
        self.layers = nn.Sequential(*layers)
        self.peq_embed_dim = output_dim

    def forward(self, x):
        assert x.ndim == 3, 'assume inputs are of size [B,T,d]'
        return self.layers(x)

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
        peq_hidden_dims,
        num_classes_per_dim,
        num_heads=8,
        num_inds=16,
        ln=False
    ):
        super().__init__()

        self.num_classes_per_dim = num_classes_per_dim
        output_dim = sum(num_classes_per_dim)

        peq_layers = []
        for dim in peq_hidden_dims:
            peq_layers.append(ISAB(input_dim, dim, num_heads, num_inds, ln))
            input_dim = dim
        self.peq_layers = nn.Sequential(*peq_layers)

        out_layers = []
        out_layers.append(nn.Linear(input_dim, output_dim))
        out_layers.append(nn.ReLU(inplace=True))
        out_layers.append(nn.Linear(output_dim, output_dim))
        self.out_layers = nn.Sequential(*out_layers)

    def forward(self, x):
        assert x.ndim == 3, 'assume inputs are of size [B,T,d]'
        peq_embed = self.peq_layers(x)
        out = self.out_layers(peq_embed)
        logits = out.split(self.num_classes_per_dim, dim=-1)
        dist = [D.Categorical(logits=logit) for logit in logits]
        return dist