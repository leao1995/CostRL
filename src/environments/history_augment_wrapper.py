import gym
from gym import spaces
import numpy as np
from collections import deque
from copy import deepcopy

from tianshou.data import Batch

'''
graphical model:
    s0 -  s1 - s2
      \  / \  /
       a1   a2


history format:

full:        f0 f1
observed:    s0 s1
mask:        m0 m1
act:         a1 a2
'''

class HistAugWrapper(gym.Wrapper):
    def __init__(self, env, max_hist_length):
        super().__init__(env)
        self.max_hist_length = max_hist_length

        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Dict)
        hist_obs_space = spaces.Box(
            low=np.repeat(np.expand_dims(obs_space['observed'].low, axis=0), max_hist_length, axis=0), 
            high=np.repeat(np.expand_dims(obs_space['observed'].high, axis=0), max_hist_length, axis=0),
            shape=[max_hist_length]+list(obs_space['observed'].shape), 
            dtype=np.float32
        )
        hist_mask_space = spaces.Box(
            low=0, high=1, shape=[max_hist_length]+list(obs_space['mask'].shape), dtype=np.float32
        )
        hist_act_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_hist_length,), dtype=np.float32
        )
        hist_space = spaces.Dict(
            {
                'full': hist_obs_space,
                'observed': hist_obs_space,
                'mask': hist_mask_space,
                'act': hist_act_space
            }
        )
        self.observation_space = spaces.Dict(
            {
                'full': obs_space['observed'],
                'observed': obs_space['observed'],
                'mask': obs_space['mask'],
                'availability': obs_space['availability'],
                'history': hist_space
            }
        )
        self.action_space = env.action_space 

    def reset(self):
        obs = self.env.reset()

        self.hist_full = deque([
            np.zeros_like(obs['observed'])
            for _ in range(self.max_hist_length)
        ], maxlen=self.max_hist_length)
        self.hist_observed = deque([
            np.zeros_like(obs['observed'])
            for _ in range(self.max_hist_length)
        ], maxlen=self.max_hist_length)
        self.hist_mask = deque([
            np.zeros_like(obs['mask'])
            for _ in range(self.max_hist_length)
        ], maxlen=self.max_hist_length)
        self.hist_act = deque([
            -1
            for _ in range(self.max_hist_length+1)
        ], maxlen=self.max_hist_length)

        self.last_full = deepcopy(self.env.state)
        self.last_observed = deepcopy(obs['observed'])
        self.last_mask = deepcopy(obs['mask'])

        obs['hist'] = Batch(
            full=np.array(self.hist_full),
            observed=np.array(self.hist_observed),
            mask=np.array(self.hist_mask),
            action=np.array(self.hist_act)
        )
        obs['full'] = deepcopy(self.env.state)

        return obs

    def _append(self, full, observed, mask, action):
        self.hist_full.append(full)
        self.hist_observed.append(observed)
        self.hist_mask.append(mask)
        self.hist_act.append(action)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not info['is_acquisition']:
            self._append(self.last_full, self.last_observed, self.last_mask, info['tsk_action'])
        self.last_full = deepcopy(self.env.state)
        self.last_observed = deepcopy(obs['observed'])
        self.last_mask = deepcopy(obs['mask'])

        obs['hist'] = Batch(
            full=np.array(self.hist_full),
            observed=np.array(self.hist_observed),
            mask=np.array(self.hist_mask),
            action=np.array(self.hist_act)
        )
        obs['full'] = deepcopy(self.env.state)

        return obs, reward, done, info