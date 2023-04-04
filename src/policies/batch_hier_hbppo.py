from .common.actor import CatPHistActor
from .common.critic import CatPHistCritic
from .common.actor_critic import ActorCritic

def build_afa_policy(config, num_embeddings, num_afa_actions):
    actor = CatPHistActor(config, num_embeddings, num_afa_actions, False)
    critic = CatPHistCritic(config, num_embeddings)
    return ActorCritic(actor, critic)

def build_tsk_policy(config, num_embeddings, num_tsk_actions):
    actor = CatPHistActor(config, num_embeddings, num_tsk_actions, False)
    critic = CatPHistCritic(config, num_embeddings)
    return ActorCritic(actor, critic)