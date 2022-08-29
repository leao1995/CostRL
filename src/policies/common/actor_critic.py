import torch
import torch.nn as nn
import torch.jit as jit

from tianshou.data import Batch

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def set_temperature(self, temp):
        self.actor.set_temperature(temp)

    def forward(self, obs):
        dist = self.actor(obs)
        if self.training:
            act = dist.sample()
        elif hasattr(dist, 'logits'):
            act = dist.logits.argmax(-1)
        elif hasattr(dist, 'probs'):
            act = dist.probs.argmax(-1)
        logp = dist.log_prob(act)
        vpred = self.critic(obs)

        return Batch(dist=dist, act=act, policy=Batch(logp=logp, vpred=vpred))