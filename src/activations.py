import numpy as np
import math


def linear(x):
    return x

def sigmoid(x):
    if (x > 100):
        x = 100
    if (x < -100):
        x = -100
    return (1 / (1 + math.exp(-4 * x)) - 0.5) * 2

def neg_sigmoid(x):
    if (x > 100):
        x = 100
    if (x < -100):
        x = -100
    return 1 - (1 / (1 + math.exp(-(10*x-5))))

def relu(x):
    return (abs(x) + x) / 2

ACTS = {
    0: linear,
    1: sigmoid,
    2: neg_sigmoid
    }

def get_act(num):
    return ACTS[num]
