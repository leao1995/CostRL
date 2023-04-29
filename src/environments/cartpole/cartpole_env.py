import gym
from gym.utils import seeding

class CartPoleEnv(gym.Env):
    def __init__(self):
        self.seed()
        self.mdp = gym.make("CartPole-v1")
        self.action_space = self.mdp.action_space
        self.observation_space = self.mdp.observation_space

        self.num_observable_features = 4
        self.num_measurable_features = 4
        self.measurable_feature_ids = [0, 1, 2, 3]

        self.observation_type = "Continuous"
        self.action_type = "Discrete"

    def step(self, action):
        return self.mdp.step(action)
    
    def reset(self):
        return self.mdp.reset()

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]