import numpy as np
from gym.utils import seeding
from gym import spaces
from scipy.spatial.distance import hamming

class MNIST_ACC:

    def __init__(self):
        data = np.genfromtxt('src/tasks/mnist_train.csv', delimiter=',')
        self.X = data[:,1:] / 255
        self.Y = data[:,0]
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,))

    def reset(self):
        return self.X

    def step(self, action):
        ans = np.argsort(action)[:,-1]
        reward = 1 - hamming(ans, self.Y)
        return [], reward, True, {}

    def seed(self, seed=None):
        return


    def close(self):
        return

class MNIST_MSE:

    def __init__(self):
        data = np.genfromtxt('src/tasks/mnist_train.csv', delimiter=',')
        self.X = data[:,1:]
        self.Y = data[:,0]

    def reset(self):
        return self.X

    def step(self, action):
        ans = np.argsort(action)[:,-1]
        reward = 1 - hamming(ans, self.Y)
        return [], reward, True, {}

    def seed(self, seed=None):
        return


    def close(self):
        return
