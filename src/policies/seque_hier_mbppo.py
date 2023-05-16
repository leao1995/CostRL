from .common.actor import BeliefSetActor, BeliefEnsActor
from .common.critic import BeliefSetCritic, BeliefEnsCritic
from .common.actor_critic import ActorCritic

class PolicyBuilder:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def build_afa_policy(self):
        # environment specific hyperparameters
        belief_dim = self.config.belief_dim
        num_afa_actions = self.env.num_measurable_features + 1
        num_tsk_actions = self.env.action_space.n

        if self.config.belief_embed_type == "set":
            actor = BeliefSetActor(self.config, belief_dim, num_afa_actions, True)
            critic = BeliefSetCritic(self.config, belief_dim)
        elif self.config.belief_embed_type == "ensemble":
            actor = BeliefEnsActor(self.config, belief_dim, num_afa_actions, True)
            critic = BeliefEnsCritic(self.config, belief_dim)
        return ActorCritic(actor, critic)

    def build_tsk_policy(self):
        # environment specific hyperparameters
        belief_dim = self.config.belief_dim
        num_afa_actions = self.env.num_measurable_features + 1
        num_tsk_actions = self.env.action_space.n

        if self.config.belief_embed_type == "set":
            actor = BeliefSetActor(self.config, belief_dim, num_tsk_actions, False)
            critic = BeliefSetCritic(self.config, belief_dim)
        elif self.config.belief_embed_type == "ensemble":
            actor = BeliefEnsActor(self.config, belief_dim, num_tsk_actions, False)
            critic = BeliefEnsCritic(self.config, belief_dim)
        return ActorCritic(actor, critic)