from .common.actor import BeliefSetActor, BeliefEnsActor
from .common.critic import BeliefSetCritic, BeliefEnsCritic
from .common.actor_critic import ActorCritic

def build_afa_policy(config, belief_dim, num_afa_actions):
    if config.belief_embed_type == "set":
        actor = BeliefSetActor(config, belief_dim, num_afa_actions, True)
        critic = BeliefSetCritic(config, belief_dim)
    elif config.belief_embed_type == "ensemble":
        actor = BeliefEnsActor(config, belief_dim, num_afa_actions, True)
        critic = BeliefEnsCritic(config, belief_dim)
    return ActorCritic(actor, critic)

def build_tsk_policy(config, belief_dim, num_tsk_actions):
    if config.belief_embed_type == "set":
        actor = BeliefSetActor(config, belief_dim, num_tsk_actions, False)
        critic = BeliefSetCritic(config, belief_dim)
    elif config.belief_embed_type == "ensemble":
        actor = BeliefEnsActor(config, belief_dim, num_tsk_actions, False)
        critic = BeliefEnsCritic(config, belief_dim)
    return ActorCritic(actor, critic)