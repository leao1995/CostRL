import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.networks.embed_pool import EmbeddingPool
from src.networks.mlp import MLP
from src.networks.xformer import ISAB, PMA

class CatObsActor(nn.Module):
    def __init__(self, config, num_embeddings, num_actions, logit_constraint):
        super().__init__()

        self.cat_embed = EmbeddingPool(num_embeddings, config.categorical_embed_dim)
        embed_dim = self.cat_embed.output_dim
        self.actor = MLP(embed_dim, config.actor_layers, num_actions)

        self.temperature = 1.0

        assert logit_constraint is False, 'fully observed actor do not need to postprocess logits'

    def set_temperature(self, temp):
        self.temperature = temp

    def forward(self, obs):
        embed = self.cat_embed(obs.long())
        logits = self.actor(embed) / self.temperature
        dist = Categorical(logits=logits)
        return dist

class CatPObsActor(nn.Module):
    def __init__(self, config, num_embeddings, num_actions, logit_constraint):
        super().__init__()

        self.cat_embed = EmbeddingPool(num_embeddings, config.categorical_embed_dim)
        embed_dim = self.cat_embed.output_dim
        self.actor = MLP(embed_dim, config.actor_layers, num_actions)

        self.temperature = 1.0

        self.logit_constraint = logit_constraint

    def set_temperature(self, temp):
        self.temperature = temp

    def forward(self, obs):
        cat_input = (obs.observed + 1) * obs.mask # 0 means unobserved
        embed = self.cat_embed(cat_input.long())
        logits = self.actor(embed) / self.temperature
        if self.logit_constraint:
            availability = obs.availability.bool()
            min_value = torch.tensor(-1e12).to(logits)
            logits = torch.where(availability, logits, min_value)
        dist = Categorical(logits=logits)
        return dist

class BeliefSetActor(nn.Module):
    def __init__(self, config, belief_dim, num_actions, logit_constraint):
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
        self.actor = MLP(embed_dim, config.actor_layers, num_actions)

        self.temperature = 1.0

        self.logit_constraint = logit_constraint

    def set_temperature(self, temp):
        self.temperature = temp

    def forward(self, obs):
        assert obs.belief.ndim == 3
        embed = self.embed_net(obs.belief)
        logits = self.actor(embed) / self.temperature
        if self.logit_constraint:
            availability = obs.availability.bool()
            min_value = torch.tensor(-1e12).to(logits)
            logits = torch.where(availability, logits, min_value)
        dist = Categorical(logits=logits)
        return dist