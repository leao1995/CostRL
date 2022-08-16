from .common.actor import BeliefSetActor
from .common.critic import BeliefSetCritic
from .common.actor_critic import ActorCritic

def build_afa_policy(config, belief_dim, num_afa_actions):
    actor = BeliefSetActor(config, belief_dim, num_afa_actions, True)
    critic = BeliefSetCritic(config, belief_dim)
    return ActorCritic(actor, critic)

def build_tsk_policy(config, belief_dim, num_tsk_actions):
    actor = BeliefSetActor(config, belief_dim, num_tsk_actions, False)
    critic = BeliefSetCritic(config, belief_dim)
    return ActorCritic(actor, critic)