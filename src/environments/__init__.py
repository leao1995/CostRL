import gym
from gym.envs.registration import register

register(
    id='sepsis-v0',
    entry_point='src.environments.sepsis.sepsis_env:SepsisEnv'
)

from .episode_length_wrapper import EpisodeLengthWrapper

def get_environment(config):
    env = gym.make(config.name)
    if hasattr(config, 'max_episode_length') and config.max_episode_length > 0:
        env = EpisodeLengthWrapper(env, config.max_episode_length)

    return env