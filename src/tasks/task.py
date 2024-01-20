import gym
from gym import spaces
from tasks.retina import Retina
from tasks.retinaNM import RetinaNM
from tasks.parity import Parity, ParitySmooth
from tasks.addition import Addition
from tasks.mnist import MNIST_ACC, MNIST_MSE
from tasks.linkage import Linkage
import numpy as np
import math

def get_task(name, config):
    gym.logger.set_level(40)
    if (name == 'Retina'):
        return Retina()
    elif (name == 'RetinaNM'):
        return RetinaNM()
    elif (name == 'Parity'):
        return Parity(config.input_size)
    elif (name == 'ParitySmooth'):
        return ParitySmooth(config.input_size)
    elif (name == 'MNIST_ACC'):
        return MNIST_ACC()
    elif (name == 'MNIST_MSE'):
        return MNIST_MSE()
    elif (name == 'Add'):
        return Addition()
    elif (name == 'Linkage'):
        return Linkage(config.input_size)
    else:
        env = gym.make(name)
        if name == 'BipedalWalker-v3':
            low = np.array(
                [
                    -math.pi,
                    -5,
                    -5,
                    -5,
                    -math.pi,
                    -5,
                    -math.pi,
                    -5,
                    -0.0,
                    -math.pi,
                    -5,
                    -math.pi,
                    -5,
                    -0.0,
                ]
                + [-1] * 10
            ).astype(np.float32)
            high = -low
            env.observation_space = spaces.Box(low, high)
        return env
