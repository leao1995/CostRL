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
        self.terminal_act = self.num_measurable_features

        self.observation_space = spaces.Dict(
            {
                'observed': spaces.Box(-np.inf, np.inf, (self.num_observable_features,), np.float32),
                'mask': spaces.Box(0, 1, (self.num_observable_features,), np.float32),
                'availability': spaces.Box(0, 1, (self.num_measurable_features+1,), np.float32)
            }
        )
        self.action_space = spaces.Discrete(self.num_measurable_features+1)

    def _get_init_mask(self):
        init_mask = [i not in self.measurable_feature_ids for i in range(self.num_observable_features)]
        init_mask = np.array(init_mask, dtype=np.float32)

        return init_mask

    def _get_afa_mask(self):
        afa_mask = [1-self.mask[i] for i in self.measurable_feature_ids]
        afa_mask = np.array(afa_mask, dtype=np.float32)

        return afa_mask

    def _get_availability(self):
        afa_mask = self._get_afa_mask() # mask for measurable features
        availability = np.append(afa_mask, 1)
        
        return availability

    def _get_observation(self):
        observed = self.state * self.mask
        availability = self._get_availability()

        return Batch(observed=observed, mask=self.mask.copy(), availability=availability)

    def reset(self):
        self.mask = self._get_init_mask()
        return self._get_observation()

    def step(self, action):
        if action == self.terminal_act:
            return self._get_observation(), 0, True, {}
        else:
            self.mask[self.measurable_feature_ids[action]] = 1
            return self._get_observation(), -self.cost, False, {}
