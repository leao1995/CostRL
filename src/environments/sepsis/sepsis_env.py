import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from .MDP import MDP
from .Action import Action
from .State import State

class SepsisEnv(gym.Env):
    def __init__(self):
        self.seed()
        self.mdp = MDP()
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=np.array([3,3,2,5,2,2,2,2])-1, shape=(8,), dtype=np.float32)

        self.num_observable_features = 8
        self.num_measurable_features = 4
        self.measurable_feature_ids = [0, 1, 2, 3]

    def step(self, action):
        done = False
        reward = self.mdp.transition(Action(action_idx=action))
        if reward != 0:
            done = True
        
        state_vec = self.mdp.state.get_state_vector()
        state_vec = np.append(state_vec, self.mdp.state.diabetic_idx)
        state_vec = state_vec.astype(np.float32)

        return state_vec, reward, done, {}

    def reset(self):
        self.mdp.state = self.mdp.get_new_state()
        state_vec = self.mdp.state.get_state_vector()
        state_vec = np.append(state_vec, self.mdp.state.diabetic_idx)

        return state_vec.astype(np.float32)

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]
