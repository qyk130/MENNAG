import numpy as np
from gym.utils import seeding
from gym import spaces
from scipy.stats import norm, beta

class Addition():

    def __init__(self, data_size=1024):
        self.data_size = data_size
        self.data = []
        self.current = 0
        self.action_space = spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)

    def reset(self):
        c = 0.9
        Rho = np.array([[1, c, c, c],
                        [c, 1, c, c],
                        [c, c, 1, c],
                        [c, c, c, 1]])
        Z1 = self.np_random.multivariate_normal([0,0,0,0], Rho, self.data_size)
        Z2 = self.np_random.multivariate_normal([0,0,0,0], Rho, self.data_size)
        B1 = beta.ppf(norm.cdf(Z1),0.5, 0.5) * 2 - 1
        B2 = beta.ppf(norm.cdf(Z2),0.5, 0.5) * 2 - 1
        self.Y = np.mean(B1,axis=1) * np.mean(B2,axis=1)
        for i in range(self.data_size):
            input = np.zeros(8)
            input[0:4] = B1[i]
            input[4:8] = B2[i]
            self.data.append(input)
        self.current = 0
        return self.data[0]

    def step(self, action):
        self.current += 1
        input = self.data[self.current]
        reward = -abs(self.Y[self.current] - action)
        if (self.current == self.data_size - 1):
            done = True
        else:
            done = False
        return self.data[self.current], reward, done, {}


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def close(self):
        return
