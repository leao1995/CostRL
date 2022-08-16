from nis import cat
from .common.actor import CatPObsActor
from .common.critic import CatPObsCritic
from .common.actor_critic import ActorCritic

def build_afa_policy(config, num_embeddings, num_afa_actions):
    actor = CatPObsActor(config, num_embeddings, num_afa_actions, False)
    critic = CatPObsCritic(config, num_embeddings)
    return ActorCritic(actor, critic)

def build_tsk_policy(config, num_embeddings, num_tsk_actions):
    actor = CatPObsActor(config, num_embeddings, num_tsk_actions, False)
    critic = CatPObsCritic(config, num_embeddings)
    return ActorCritic(actor, critic)