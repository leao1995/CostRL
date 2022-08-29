import gym
from gym import spaces
import numpy as np
from copy import deepcopy

from tianshou.data import Batch

class BatchAFAWrapper(gym.Wrapper):
    def __init__(self, env, cost):
        super().__init__(env)
        self.cost = cost

        self.num_observable_features = env.unwrapped.num_observable_features
        self.num_measurable_features = env.unwrapped.num_measurable_features
        self.measurable_feature_ids = env.unwrapped.measurable_feature_ids

        self.num_afa_actions = 2 ** self.num_measurable_features
        self.num_tsk_actions = env.unwrapped.action_space.n

        self.observation_space = spaces.Dict(
            {
                'observed': env.unwrapped.observation_space,
                'mask': spaces.Box(0, 1, env.unwrapped.observation_space.shape),
            }
        )
        self.action_space = spaces.Dict(
            {
                'afa_action': spaces.Discrete(self.num_afa_actions),
                'tsk_action': env.unwrapped.action_space
            }
        )

    def _get_init_mask(self):
        init_mask = [i not in self.measurable_feature_ids for i in range(self.num_observable_features)]
        init_mask = np.array(init_mask, dtype=np.float32)

        return init_mask

    def _get_observation(self):
        observed = self.state * self.mask

        return Batch(observed=observed, mask=self.mask.copy())

    def reset(self):
        self.state = self.env.reset()
        self.mask = self._get_init_mask()

        self.acquired = False

        return self._get_observation()

    def _parse_action(self, action):
        afa_action = action['afa_action']
        tsk_action = action['tsk_action']
        if self.acquired:
            is_acquisition = False
            afa_action = None
        else:
            is_acquisition = True
            tsk_action = None

        return is_acquisition, afa_action, tsk_action

    def _acquire(self, action):
        afa_mask = list(map(int, list(bin(action+self.num_afa_actions)[3:])))
        num_acquisitions = np.sum(afa_mask)
        afa_idx = [id for i, id in enumerate(self.measurable_feature_ids) if afa_mask[i] == 1]
        self.mask[afa_idx] = 1

        return num_acquisitions

    def step(self, action):
        is_acquisition, afa_action, tsk_action = self._parse_action(action)
        if is_acquisition:
            num_acquisitions = self._acquire(afa_action)
            reward, done, info = -self.cost * num_acquisitions, False, {}
            self.acquired = True
        else:
            self.state, reward, done, info = self.env.step(tsk_action)
            self.mask = self._get_init_mask()
            self.acquired = False

        info['is_acquisition'] = is_acquisition
        info['task_reward'] = 0 if is_acquisition else reward
        info['num_acquisitions'] = num_acquisitions if is_acquisition else 0
        info['tsk_action'] = None if is_acquisition else tsk_action
        info['afa_action'] = afa_action if is_acquisition else None
        info['fully_observed'] = deepcopy(self.state)

        return self._get_observation(), reward, done, info
