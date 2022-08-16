from .common.actor import BeliefSetActor
from .common.critic import BeliefSetCritic
from .common.actor_critic import ActorCritic

def build_policy(config, belief_dim, num_actions):
    actor = BeliefSetActor(config, belief_dim, num_actions, True)
    critic = BeliefSetCritic(config, belief_dim)
    return ActorCritic(actor, critic)