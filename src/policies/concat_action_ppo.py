from .common.actor import CatPObsActor
from .common.critic import CatPObsCritic
from .common.actor_critic import ActorCritic

def build_policy(config, num_embeddings, num_actions):
    actor = CatPObsActor(config, num_embeddings, num_actions, True)
    critic = CatPObsCritic(config, num_embeddings)
    return ActorCritic(actor, critic)