from .common.actor import CatPObsActor, ConPObsActor
from .common.critic import CatPObsCritic, ConPObsCritic
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

            actor = CatPObsActor(self.config, num_embeddings, num_actions, True)
            critic = CatPObsCritic(self.config, num_embeddings)
            return ActorCritic(actor, critic)

        if self.env.observation_type == "Continuous" and self.env.action_type == "Discrete":
            # environment specific hyperparameters
            obs_dim = self.env.observation_space.shape[0]
            num_actions = self.env.num_measurable_features + self.env.action_space.n

            actor = ConPObsActor(self.config, obs_dim, num_actions, True)
            critic = ConPObsCritic(self.config, obs_dim)
            return ActorCritic(actor, critic)
    