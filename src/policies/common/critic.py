import torch
import torch.nn as nn

from src.networks.embed_pool import EmbeddingPool
from src.networks.mlp import MLP
from src.networks.xformer import ISAB, PMA

class CatObsCritic(nn.Module):
    def __init__(self, config, num_embeddings):
        super().__init__()

        self.cat_embed = EmbeddingPool(num_embeddings, config.categorical_embed_dim)
        embed_dim = self.cat_embed.output_dim
        self.critic = MLP(embed_dim, config.critic_layers, 1)

    def forward(self, obs):
        embed = self.cat_embed(obs.long())
        values = self.critic(embed)

        return values.squeeze(dim=1)

class CatPObsCritic(nn.Module):
    def __init__(self, config, num_embeddings):
        super().__init__()

        self.cat_embed = EmbeddingPool(num_embeddings, config.categorical_embed_dim)
        embed_dim = self.cat_embed.output_dim
        self.critic = MLP(embed_dim, config.critic_layers, 1)

    def forward(self, obs):
        cat_input = (obs.observed + 1) * obs.mask # 0 means unobserved
        embed = self.cat_embed(cat_input.long())
        values = self.critic(embed)

        return values.squeeze(dim=1)

class BeliefSetCritic(nn.Module):
    def __init__(self, config, belief_dim):
        super().__init__()

        layers = []
        input_dim = belief_dim
        for dim in config.belief_embed_dims:
            layers.append(ISAB(input_dim, dim, config.num_heads, config.num_inds, config.use_ln))
            input_dim = dim
        layers.append(PMA(input_dim, config.num_heads, 1, config.use_ln))
        layers.append(nn.Flatten())
        self.embed_net  = nn.Sequential(*layers)
        embed_dim = input_dim
        self.critic = MLP(embed_dim, config.critic_layers, 1)

    def forward(self, obs):
        assert obs.belief.ndim == 3
        embed = self.embed_net(obs.belief)
        values = self.critic(embed)

        return values.squeeze(dim=1)