import numpy as np
from gym.utils import seeding
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
    def __init__(self, n, k, linkage_density=0.2):
        self.n = n
        self.k = k
        self.linkages = [[{} for _ in range(k)] for _ in range(n)]
        self.linkage_density = linkage_density

    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.seed = seed
    
    def reset(self):
        if len(self.linkages[0][0]) == 0:
            level_size = []
            for i in range(self.k):
                    level_size.append(math.comb(self.k, i + 1) * (2**(i + 1)))

            for block in range(self.n):
                key_list = []
                for i in range(self.k):
                    comb = combinations(range(self.k), i + 1)
                    bitstrings = list(generate_bitstrings(i + 1))
                    level_linkages_list = []
                    for c in comb:
                        for s in bitstrings:
                            linkage = ['.'] * self.k * self.n
                            for j in range(i + 1):
                                linkage[block * self.k + c[j]] = s[j]
                            linkage = ''.join(linkage)
                            level_linkages_list.append(linkage) 
                    if i == 0:
                        sample = level_linkages_list
                    else:
                        random.seed(self.seed)
                        sample = random.sample(level_linkages_list, math.ceil(len(level_linkages_list) * self.linkage_density))
                    strength = self.np_random.random(len(sample))
                    strength /= np.max(strength) * math.comb(self.k, i + 1)
                    self.linkages[block][i].update(zip(sample,strength))
                #print(self.linkages[block][self.k - 1])
                for block in range(self.n):
                    for i in range(self.k - 1, -1, -1):

                        for j in range(i):

                            for key in self.linkages[block][j]:
                                overlap = []
                                min_overlap = 100

                                for key2 in self.linkages[block][i]:
                                    if re.match(key, key2):
                                        overlap.append(key2)
                                        if self.linkages[block][i][key2] < min_overlap:
                                            min_overlap = self.linkages[block][i][key2]


                                if len(overlap) > 0:
                                    for key2 in overlap: 
                                        self.linkages[block][i][key2] -= min_overlap
                                    self.linkages[block][j][key] += min_overlap

            for block in range(self.n):
                for i in range(self.k):
                    #print(self.linkages[block][i])
                    zeros = []
                    for key in self.linkages[block][i]:
                        if self.linkages[block][i][key] == 0:
                            zeros.append(key)
                    for key in zeros:
                        del self.linkages[block][i][key]
                    #print(len(self.linkages[block][i]))

            for block in range(self.n):
                for i in range(self.k):
                    
                    sum_d = sum(self.linkages[block][i].values())
                    max_d = max(self.linkages[block][i].values())
                    
                    p = len(self.linkages[block][i]) / (math.comb(self.k, i + 1) * 2**(i + 1))
                    print(p)
                    if i == self.k - 1:
                        p = 1
                    c = math.comb(self.k, i + 1) *0.4 + 0.6
                        
                    for key in self.linkages[block][i]:
                        self.linkages[block][i][key] /= max_d * c * p

                    
                    
                    print(i,math.comb(self.k, i + 1), sum(self.linkages[block][i].values()) / len(self.linkages[block][i].values()), sum(self.linkages[block][i].values()))

    
    def step(self, input):
        # The input is a bitstring of length self.size, for each linkage in the form of regular expression, 
        # if the input matches the regular expression, the linkage is activated and the value is added to the output.
        reward = 0
        info = {}
        if not isinstance(input, str):
            input = ''.join(str(x) for x in input)
        linkage_utilization = [0] * self.k
        for block in range(self.n):
            block_reward = 0
            for i in range(self.k):
                for linkage in self.linkages[block][i]:
                    if re.match(linkage, input):
                        block_reward += self.linkages[block][i][linkage]
                        linkage_utilization[i] += self.linkages[block][i][linkage]
            reward += block_reward
        info['linkage_utilization'] = linkage_utilization
        return reward, info
        #return reward

    def close(self):
        return