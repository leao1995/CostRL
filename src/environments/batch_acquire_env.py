import gym
from gym import spaces
import numpy as np

from tianshou.data import Batch

class AcquireEnv(gym.Env):
    def __init__(self, env, state, cost):
        self.state = state.copy()
        self.cost = cost
        self.num_observable_features = env.num_observable_features
        self.num_measurable_features = env.num_measurable_features
        self.measurable_feature_ids = env.measurable_feature_ids

        self.num_actions = 2 ** self.num_measurable_features

        self.observation_space = spaces.Dict(
            {
                'observed': spaces.Box(-np.inf, np.inf, (self.num_observable_features,), np.float32),
                'mask': spaces.Box(0, 1, (self.num_observable_features,), np.float32),
            }
        )
        self.action_space = spaces.Discrete(self.num_actions)

    def _get_init_mask(self):
        init_mask = [i not in self.measurable_feature_ids for i in range(self.num_observable_features)]
        init_mask = np.array(init_mask, dtype=np.float32)

        return init_mask

    def _get_observation(self):
        observed = self.state * self.mask

        return Batch(observed=observed, mask=self.mask.copy())

    def _acquire(self, action):
        afa_mask = list(map(int, list(bin(action+self.num_actions)[3:])))
        num_acquisitions = np.sum(afa_mask)
        afa_idx = [id for i, id in enumerate(self.measurable_feature_ids) if afa_mask[i] == 1]
        self.mask[afa_idx] = 1

        return num_acquisitions

    def reset(self):
        self.mask = self._get_init_mask()
        return self._get_observation()

    def step(self, action):
        num_acquisitions = self._acquire(action)
        reward = -self.cost * num_acquisitions
        info = {'num_acquisitions': num_acquisitions}
        
        return self._get_observation(), reward, True, info
