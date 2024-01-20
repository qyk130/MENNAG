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
        self.linkages = [{} for _ in range(10)]
        self.size = size
        self.linkage_density = linkage_density

    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
    
    def reset(self):
        if len(self.linkages[0]) == 0:
            level_size = []
            level_strength = []
            for i in range(self.size):
                level_size.append(math.comb(self.size, i + 1) * (2**(i + 1)))
                level_strength.append(math.comb(self.size, i + 1))
            level_strength = np.array(level_strength)
            level_strength = 1 / level_strength
            for i in range(self.size):
                linkage = ['.'] * self.size
                linkage[i] = '0'
                linkage = ''.join(linkage)
                self.linkages[0][linkage] = self.np_random.uniform(0, level_strength[0])
                linkage = ['.'] * self.size
                linkage[i] = '1'
                linkage = ''.join(linkage)
                self.linkages[0][linkage] = self.np_random.uniform(0, level_strength[0])
            for i in range(1, self.size):
                comb = combinations(range(self.size), i + 1)
                bitstrings = list(generate_bitstrings(i + 1))
                level_linkages = {}
                for c in comb:
                    for b in bitstrings:
                        linkage = ['.'] * self.size
                        for j in range(i + 1):
                            linkage[c[j]] = b[j]
                        linkage = ''.join(linkage)
                        level_linkages[linkage] = self.np_random.uniform(0, level_strength[i])
                sample = random.sample(list(level_linkages.items()), math.ceil(level_size[i] * self.linkage_density))
                self.linkages[i].update(sample)
    
    def step(self, input):
        # The input is a bitstring of length self.size, for each linkage in the form of regular expression, 
        # if the input matches the regular expression, the linkage is activated and the value is added to the output.
        reward = 0
        info = {}
        if not isinstance(input, str):
            input = map(str, input) 
        linkage_utilization = [0] * self.size
        for i in range(self.size):
            for linkage in self.linkages[i]:
                if re.match(linkage, ''.join(input)):
                    reward += self.linkages[i][linkage]
                    linkage_utilization[i] += self.linkages[i][linkage]
        info['linkage_utilization'] = linkage_utilization
        return input, reward, True, info
        #return reward

    def close(self):
        return