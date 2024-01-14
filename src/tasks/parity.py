import numpy as np
from gym.utils import seeding
from gym import spaces

def bin_to_int(bits):
    l = len(bits)
    result = 0
    for i in range(l):
        result += bits[i] << (l - i - 1)
    return result

class Parity:

    def __init__(self, input_size):
        self.state = [0] * input_size
        self.current = 0
        self.input_size = input_size

    def reset(self):
        self.current = 0
        self.state = [0] * self.input_size
        return self.get_state()

    def step(self, action):
        s = self.get_state()
        sum = 0
        for c in s:
            sum += int(c)
        self.current += 1
        if (sum % 2 == action):
            reward = 1 / 2**self.input_size
        else:
            reward = 0
        if (self.current == 2**self.input_size - 1):
            done = True
        else:
            done = False

        s = self.get_state()

        return s, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def get_state(self):
        s = np.binary_repr(self.current, width = self.input_size)
        for i in range(self.input_size):
            n = int(s[i])
            self.state[i] = n
        return np.array(self.state)

    def close(self):
        return

class ParitySmooth:

    def __init__(self, input_size):
        self.state = np.zeros(input_size, dtype=int)
        self.current = 0
        self.input_size = input_size
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self):
        self.current = 0
        return self.get_state()

    def step(self, action):
        s = self.get_state()
        sum = 0
        for c in s:
            sum += int(c)
        self.current += 1
        if (sum % 2 == 1):
            reward = -((1 - action)**2 / 2**self.input_size)
        else:
            reward = -((-1 - action)**2 / 2**self.input_size)
        if (self.current == 2**self.input_size):
            done = True
        else:
            done = False

        s = self.get_state()

        return s, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def get_state(self):
        s = np.binary_repr(self.current, width = self.input_size)
        for i in range(self.input_size):
            self.state[i] = int(s[i])
        return self.state

    def close(self):
        return
