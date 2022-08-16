import gym

class EpisodeLengthWrapper(gym.Wrapper):
    def __init__(self, env, max_length):
        super().__init__(env)
        self.max_length = max_length

    def reset(self):
        obs = self.env.reset()
        self.length = 0

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.length += 1
        if self.length >= self.max_length:
            done = True

        return obs, reward, done, info