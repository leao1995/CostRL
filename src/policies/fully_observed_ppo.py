from .common.actor import CatObsActor, ConObsActor
from .common.critic import CatObsCritic, ConObsCritic
from .common.actor_critic import ActorCritic

class PolicyBuilder:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def build_policy(self):
        if self.env.observation_type == "Categorical" and self.env.action_type == "Discrete":
            # environment specific hyperparameters
            obs_high = self.env.observation_space.high
            num_embeddings = list(map(int, obs_high + 1))
            num_actions = self.env.action_space.n

            actor = CatObsActor(self.config, num_embeddings, num_actions, False)
            critic = CatObsCritic(self.config, num_embeddings)
            return ActorCritic(actor, critic)
        
        if self.env.observation_type == "Continuous" and self.env.action_type == "Discrete":
            # environment specific hyperparameters
            obs_dim = self.env.observation_space.shape[0]
            num_actions = self.env.action_space.n

            actor = ConObsActor(self.config, obs_dim, num_actions, False)
            critic = ConObsCritic(self.config, obs_dim)
            return ActorCritic(actor , critic)
