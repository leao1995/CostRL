from .common.actor import CatObsActor
from .common.critic import CatObsCritic
from .common.actor_critic import ActorCritic

def build_policy(config, num_embeddings, num_actions):
    actor = CatObsActor(config, num_embeddings, num_actions, False)
    critic = CatObsCritic(config, num_embeddings)
    return ActorCritic(actor, critic)