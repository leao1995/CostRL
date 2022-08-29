import gym
from gym import spaces
import numpy as np
from copy import deepcopy

class ConcatAFAWrapper(gym.Wrapper):
    def __init__(self, env, cost):
        super().__init__(env)
        self.cost = cost

        self.num_observable_features = env.unwrapped.num_observable_features
        self.num_measurable_features = env.unwrapped.num_measurable_features
        self.measurable_feature_ids = env.unwrapped.measurable_feature_ids
        self.num_actions = env.unwrapped.action_space.n
        
        self.observation_space = spaces.Dict(
            {
                'observed': env.unwrapped.observation_space,
                'mask': spaces.Box(0, 1, env.unwrapped.observation_space.shape),
                'availability': spaces.Box(0, 1, shape=(self.num_measurable_features+self.num_actions,))
            }
        )
        self.action_space = spaces.Discrete(self.num_measurable_features+self.num_actions)

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
        tsk_mask = np.ones([self.num_actions]) # task actions are always available
        availability = np.concatenate([afa_mask, tsk_mask])
        
        return availability

    def reset(self):
        self.state = self.env.reset()
        self.mask = self._get_init_mask()
        observed = self.state * self.mask

        return {
            'observed': observed,
            'mask': self.mask.copy(),
            'availability': self._get_availability()
        }

    def _parse_action(self, action):
        if action < self.num_measurable_features:
            is_acquisition = True
            afa_action = action
            tsk_action = None
        else:
            is_acquisition = False
            afa_action = None
            tsk_action = action - self.num_measurable_features
        
        return is_acquisition, afa_action, tsk_action

    def step(self, action):
        is_acquisition, afa_action, tsk_action = self._parse_action(action)
        if is_acquisition:
            self.mask[self.measurable_feature_ids[afa_action]] = 1
            reward, done, info = -self.cost, False, {}
        else:
            self.state, reward, done, info = self.env.step(tsk_action)
            self.mask = self._get_init_mask()

        info['is_acquisition'] = is_acquisition
        info['task_reward'] = 0 if is_acquisition else reward
        info['num_acquisitions'] = 1 if is_acquisition else 0
        info['tsk_action'] = None if is_acquisition else tsk_action
        info['afa_action'] = afa_action if is_acquisition else None
        info['fully_observed'] = deepcopy(self.state)

        obs = {
            'observed': self.state * self.mask,
            'mask': self.mask.copy(),
            'availability': self._get_availability()
        }

        return obs, reward, done, info