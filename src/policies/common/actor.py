import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.networks.embed_pool import EmbeddingPool
from src.networks.mlp import MLP
from src.networks.xformer import ISAB, PMA

## Categorical Observation

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
 
class CatPHistActor(nn.Module):
    def __init__(self, config, num_embeddings, num_actions, logit_constraint):
        super().__init__()

        self.cat_embed = EmbeddingPool(num_embeddings, config.categorical_embed_dim)
        hist_dim = self.cat_embed.output_dim
        self.embed_net = MLP(hist_dim, config.hist_embed_dims)
        embed_dim = self.embed_net.output_dim
        self.actor = MLP(embed_dim, config.actor_layers, num_actions)

        self.temperature = 1.0

        self.logit_constraint = logit_constraint

    def set_temperature(self, temp):
        self.temperature = temp

    def forward(self, obs):
        cat_input = (obs.hist.observed + 1) * obs.hist.mask # 0 means unobserved
        hist_embed = self.cat_embed(cat_input.long())
        assert hist_embed.ndim == 3
        embed = self.embed_net(hist_embed).mean(dim=1)
        logits = self.actor(embed) / self.temperature
        if self.logit_constraint:
            availability = obs.availability.bool()
            min_value = torch.tensor(-1e12).to(logits)
            logits = torch.where(availability, logits, min_value)
        dist = Categorical(logits=logits)
        return dist

## Continuous Observation

class ConObsActor(nn.Module):
    def __init__(self, config, observation_dim, num_actions, logit_constraint):
        super().__init__()

        self.actor = MLP(observation_dim, config.actor_layers, num_actions)

        self.temperature = 1.0

        assert logit_constraint is False, 'fully observed actor do not need to postprocess logits'

    def set_temperature(self, temp):
        self.temperature = temp

    def forward(self, obs):
        logits = self.actor(obs) / self.temperature
        dist = Categorical(logits=logits)
        return dist
    
class ConPObsActor(nn.Module):
    def __init__(self, config, observation_dim, num_actions, logit_constraint):
        super().__init__()

        embed_dim = observation_dim * 2
        self.actor = MLP(embed_dim, config.actor_layers, num_actions)

        self.temperature = 1.0

        self.logit_constraint = logit_constraint

    def set_temperature(self, temp):
        self.temperature = temp

    def forward(self, obs):
        embed = torch.cat([obs.observed, obs.mask], dim=1)
        logits = self.actor(embed) / self.temperature
        if self.logit_constraint:
            availability = obs.availability.bool()
            min_value = torch.tensor(-1e12).to(logits)
            logits = torch.where(availability, logits, min_value)
        dist = Categorical(logits=logits)
        return dist
    
class ConPHistActor(nn.Module):
    def __init__(self, config, obsrvation_dim, num_actions, logit_constraint):
        super().__init__()

        hist_dim = observation_dim * 2
        self.embed_net = MLP(hist_dim, config.hist_embed_dims)
        embed_dim = self.embed_net.output_dim
        self.actor = MLP(embed_dim, config.actor_layers, num_actions)

        self.temperature = 1.0

        self.logit_constraint = logit_constraint

    def set_temperature(self, temp):
        self.temperature = temp

    def forward(self, obs):
        hist_embed = torch.cat([obs.hist.observed, obs.hist.mask], dim=2)
        embed = self.embed_net(hist_embed).mean(dim=1)
        logits = self.actor(embed) / self.temperature
        if self.logit_constraint:
            availability = obs.availability.bool()
            min_value = torch.tensor(-1e12).to(logits)
            logits = torch.where(availability, logits, min_value)
        dist = Categorical(logits=logits)
        return dist

## Belief Set

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

class BeliefEnsActor(nn.Module):
    def __init__(self, config, belief_dim, num_actions, logit_constraint):
        super().__init__()

        self.embed_net = MLP(belief_dim, config.belief_embed_dims)
        embed_dim = self.embed_net.output_dim
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
            availability = obs.availability.bool().unsqueeze(dim=1)
            min_value = torch.tensor(-1e12).to(logits)
            logits = torch.where(availability, logits, min_value)
        probs = torch.softmax(logits, dim=2).mean(dim=1)
        dist = Categorical(probs=probs)
        return dist