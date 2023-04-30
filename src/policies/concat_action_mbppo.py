from .common.actor import BeliefSetActor, BeliefEnsActor
from .common.critic import BeliefSetCritic, BeliefEnsCritic
from .common.actor_critic import ActorCritic

class PolicyBuilder:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def build_policy(self):
        # environment specific hyperparameters
        obs_high = self.env.observation_space.high
        num_embeddings = list(map(int, obs_high + 2))
        belief_dim = self.config.belief_dim
        num_actions = self.env.num_measurable_features + self.env.action_space.n

        if self.config.belief_embed_type == "set":
            actor = BeliefSetActor(self.config, belief_dim, num_actions, True)
            critic = BeliefSetCritic(self.config, belief_dim)
        elif self.config.belief_embed_type == "ensemble":
            actor = BeliefEnsActor(self.config, belief_dim, num_actions, True)
            critic = BeliefEnsCritic(self.config, belief_dim)
        return ActorCritic(actor, critic)