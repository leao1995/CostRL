import gym
from gym import spaces
import numpy as np
from copy import deepcopy

class JointAFAWrapper(gym.Wrapper):
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
            }
        )
        self.num_acquisition_actions = 2 ** self.num_measurable_features
        self.action_space = spaces.Discrete(self.num_acquisition_actions * self.num_actions)

    def _get_init_mask(self):
        init_mask = [i not in self.measurable_feature_ids for i in range(self.num_observable_features)]
        init_mask = np.array(init_mask, dtype=np.float32)

        return init_mask

    def reset(self):
        self.state = self.env.reset()
        self.mask = np.ones([self.num_observable_features], dtype=np.float32)
        observed = self.state * self.mask
        self.init = True

        return {
            'observed': observed,
            'mask': self.mask.copy(),
        }

    def _parse_action(self, action):
        tsk_action = action // self.num_acquisition_actions
        afa_action = action % self.num_acquisition_actions
        
        return tsk_action, afa_action
    
    def _acquire(self, action):
        afa_mask = list(map(int, list(bin(action+self.num_acquisition_actions)[3:])))
        num_acquisitions = np.sum(afa_mask)
        afa_idx = [id for i, id in enumerate(self.measurable_feature_ids) if afa_mask[i] == 1]
        self.mask = self._get_init_mask() 
        self.mask[afa_idx] = 1

        return num_acquisitions

    def step(self, action):
        tsk_action, afa_action = self._parse_action(action)
        self.state, reward, done, info = self.env.step(tsk_action)
        num_acquisitions = self._acquire(afa_action)
        total_reward = reward - self.cost * num_acquisitions

        info['task_reward'] = reward
        info['num_acquisitions'] = num_acquisitions + self.num_measurable_features if self.init else num_acquisitions
        info['tsk_action'] = tsk_action
        info['afa_action'] = afa_action
        info['fully_observed'] = deepcopy(self.state)
        self.init = False

        obs = {
            'observed': self.state * self.mask,
            'mask': self.mask.copy(),
        }

        return obs, total_reward, done, info