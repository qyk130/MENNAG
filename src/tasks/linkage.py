import numpy as np
from gym.utils import seeding
from gym import spaces
import math
import random
import re
from itertools import combinations

def generate_bitstrings(length):
    """Generate all possible states of a bitstring of the given length."""
    format_string = '{:0' + str(length) + 'b}'  # Format string for binary representation
    for i in range(2**length):
        yield format_string.format(i)

class Linkage:
    def __init__(self, size, linkage_density=0.2):
        self.linkages = {}
        self.size = size
        self.linkage_density = linkage_density

    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
    
    def reset(self):
        if len(self.linkages) == 0:
            sizes = 0
            for i in range(self.size):
                linkage = ['.'] * self.size
                linkage[i] = '0'
                linkage = ''.join(linkage)
                self.linkages[linkage] = self.np_random.uniform(0, 1)
                linkage = ['.'] * self.size
                linkage[i] = '1'
                linkage = ''.join(linkage)
                self.linkages[linkage] = self.np_random.uniform(0, 1)
            for i in range(2, self.size + 1):
                level_size=math.comb(self.size, i) * (2**(i + 1))
                comb = combinations(range(self.size), i)
                bitstrings = list(generate_bitstrings(i))
                level_linkages = {}
                for c in comb:
                    for b in bitstrings:
                        linkage = ['.'] * self.size
                        for j in range(i):
                            linkage[c[j]] = b[j]
                        linkage = ''.join(linkage)
                        level_linkages[linkage] = self.np_random.uniform(0, 1)
                sample = random.sample(list(level_linkages.items()), math.ceil(level_size * self.linkage_density))
                self.linkages.update(sample)
    
    def step(self, input):
        # The input is a bitstring of length self.size, for each linkage in the form of regular expression, 
        # if the input matches the regular expression, the linkage is activated and the value is added to the output.
        reward = 0
        for linkage in self.linkages:
            if re.match(linkage, input):
                reward += self.linkages[linkage]
        return input, reward, True, {}
