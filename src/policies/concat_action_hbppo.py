from .common.actor import CatPHistActor, ConPHistActor
from .common.critic import CatPHistCritic, ConPHistCritic
from .common.actor_critic import ActorCritic

class PolicyBuilder:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def build_policy(self):
        if self.env.observation_type == "Categorical" and self.env.action_type == "Discrete":
            # environment specific hyperparameters
            obs_high = self.env.observation_space.high
            num_embeddings = list(map(int, obs_high + 2))
            num_actions = self.env.num_measurable_features + self.env.action_space.n

            actor = CatPHistActor(self.config, num_embeddings, num_actions, True)
            critic = CatPHistCritic(self.config, num_embeddings)
            return ActorCritic(actor, critic)

        if self.env.observation_type == "Continuous" and self.env.action_type == "Discrete":
            # environment specific hyperparameters
            obs_dim = self.env.observation_space.shape[0]
            num_actions = self.env.num_measurable_features + self.env.action_space.n

            actor = ConPHistActor(self.config, obs_dim, num_actions, True)
            critic = ConPHistCritic(self.config, obs_dim)
            return ActorCritic(actor, critic)