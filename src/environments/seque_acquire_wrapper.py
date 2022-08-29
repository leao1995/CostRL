import gym
from gym import spaces
import numpy as np
from copy import deepcopy

from tianshou.data import Batch

class SequeAFAWrapper(gym.Wrapper):
    def __init__(self, env, cost):
        super().__init__(env)
        self.cost = cost

        self.num_observable_features = env.unwrapped.num_observable_features
        self.num_measurable_features = env.unwrapped.num_measurable_features
        self.measurable_feature_ids = env.unwrapped.measurable_feature_ids

        self.num_afa_actions = self.num_measurable_features + 1
        self.num_tsk_actions = env.unwrapped.action_space.n

        self.observation_space = spaces.Dict(
            {
                'observed': env.unwrapped.observation_space,
                'mask': spaces.Box(0, 1, env.unwrapped.observation_space.shape),
                'availability': spaces.Box(0, 1, shape=(self.num_afa_actions,))
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
        self.state = self.env.reset()
        self.mask = self._get_init_mask()

        self.last_is_terminal = False

        return self._get_observation()

    def _parse_action(self, action):
        afa_action = action['afa_action']
        tsk_action = action['tsk_action']
        if self.last_is_terminal:
            is_acquisition = False
            afa_action = None
            self.last_is_terminal = False
        elif afa_action < self.num_measurable_features:
            is_acquisition = True
            tsk_action = None
            self.last_is_terminal = False
        else:
            is_acquisition = True
            self.last_is_terminal = True
            afa_action = None
            tsk_action = None

        return is_acquisition, afa_action, tsk_action

    def _acquire(self, action):
        self.mask[self.measurable_feature_ids[action]] = 1

    def step(self, action):
        is_acquisition, afa_action, tsk_action = self._parse_action(action)
        is_terminal = False
        if is_acquisition and afa_action is not None:
            self._acquire(afa_action)
            reward, done, info = -self.cost, False, {}
        elif is_acquisition:
            is_terminal = True
            reward, done, info = 0, False, {}
        else:
            self.state, reward, done, info = self.env.step(tsk_action)
            self.mask = self._get_init_mask()

        info['is_acquisition'] = is_acquisition
        info['is_terminal'] = is_terminal
        info['task_reward'] = 0 if is_acquisition else reward
        info['num_acquisitions'] = 1 if is_acquisition and not is_terminal else 0
        info['tsk_action'] = None if is_acquisition else tsk_action
        info['afa_action'] = afa_action if is_acquisition else None
        info['fully_observed'] = deepcopy(self.state)

        return self._get_observation(), reward, done, info
