import torch
import torch.nn as nn

from src.networks.embed_pool import EmbeddingPool
from src.networks.mlp import MLP
from src.networks.xformer import ISAB, PMA

## Categorical Observation

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
    
class CatPHistCritic(nn.Module):
    def __init__(self, config, num_embeddings):
        super().__init__()

        self.cat_embed = EmbeddingPool(num_embeddings, config.categorical_embed_dim)
        hist_dim = self.cat_embed.output_dim
        self.embed_net = MLP(hist_dim, config.hist_embed_dims)
        embed_dim = self.embed_net.output_dim
        self.critic = MLP(embed_dim, config.critic_layers, 1)

    def forward(self, obs):
        cat_input = (obs.hist.observed + 1) * obs.hist.mask # 0 means unobserved
        hist_embed = self.cat_embed(cat_input.long())
        assert hist_embed.ndim == 3
        embed = self.embed_net(hist_embed).mean(dim=1)
        values = self.critic(embed)

        return values.squeeze(dim=1)

## Continuous Observation

class ConObsCritic(nn.Module):
    def __init__(self, config, observation_dim):
        super().__init__()

        self.critic = MLP(observation_dim, config.critic_layers, 1)

    def forward(self, obs):
        values = self.critic(obs)

        return values.squeeze(dim=1)

class ConPObsCritic(nn.Module):
    def __init__(self, config, observation_dim):
        super().__init__()

        embed_dim = observation_dim * 2
        self.critic = MLP(embed_dim, config.critic_layers, 1)

    def forward(self, obs):
        embed = torch.cat([obs.observed, obs.mask], dim=1)
        values = self.critic(embed)

        return values.squeeze(dim=1)

class ConPHistCritic(nn.Module):
    def __init__(self, config, observation_dim):
        super().__init__()

        hist_dim = observation_dim * 2
        self.embed_net = MLP(hist_dim, config.hist_embed_dims)
        embed_dim = self.embed_net.output_dim
        self.critic = MLP(embed_dim, config.critic_layers, 1)

    def forward(self, obs):
        hist_embed = torch.cat([obs.hist.observed, obs.hist.mask], dim=2)
        assert hist_embed.ndim == 3
        embed = self.embed_net(hist_embed).mean(dim=1)
        values = self.critic(embed)

        return values.squeeze(dim=1)

## Belief Set

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

class BeliefEnsCritic(nn.Module):
    def __init__(self, config, belief_dim):
        super().__init__()

        self.embed_net = MLP(belief_dim, config.belief_embed_dims)
        embed_dim = self.embed_net.output_dim
        self.critic = MLP(embed_dim, config.critic_layers, 1)

    def forward(self, obs):
        assert obs.belief.ndim == 3
        embed = self.embed_net(obs.belief)
        values = self.critic(embed)
        values = values.mean(dim=1)

        return values.squeeze(dim=1)