import numpy as np
from gym.utils import seeding

def bin_to_int(bits):
    l = len(bits)
    result = 0
    for i in range(l):
        result += bits[i] << (l - i - 1)
    return result

class RetinaNM:

    def __init__(self):
        self.seed()
        fourBits = list(range(16))
        self.state = np.zeros(8, dtype=int)
        self.current = 0

    def reset(self):
        self.current = 0
        return self.get_state()

    def step(self, action):
        s = self.get_state()
        r = sum(s)
        if ((r == 4) ^ action):
            reward = 1 / 256
        else:
            reward = 0
        if (self.current == 256):
            done = True
        else:
            done = False
        self.current += 1
        return s, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def get_state(self):
        s = np.binary_repr(self.current, width = 8)
        for i in range(8):
            self.state[i] = int(s[i])
        return self.state

    def close(self):
        return
