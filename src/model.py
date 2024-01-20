import numpy as np
import random
from nn.FC import FC
from bitstring import Bitstring
from analyzer import Q

def min_common_ancestry_dist(n1, n2):
    a1 = n1.ancestors
    a2 = n2.ancestors
    d = [10000]
    for k1 in a1.keys():
        try:
            d.append(a1[k1] + a2[k1])
        except KeyError:
            pass
    return min(d)

class Model():

    def __init__(self, config=None, ID=-1, birth_gen=-1):
        self.ID = ID
        self.config = config
        self.trajectory = []
        self.ancestors = {self.ID:0}
        self.maxDepth = 0
        self.birth_gen = birth_gen
        self.parents = [-1, -1]
        self.fit = -100000000

    def generate(self):
        return

    def compile(self):
        return

    def execute(self):
        return

    def detach(self):
        return

    def add_ancestors(self, a2=None):
        ancestors = {}
        if (a2 is None):
            ancestors.update((k, v + 1) for (k, v) in self.ancestors.items()
                    if v < self.config.max_ancestry_dist)
            ancestors[self.ID] = 0
        else:
            ancestors.update((k, v + 1) for (k, v) in self.ancestors.items()
                    if v < self.config.max_ancestry_dist)
            for k, v in a2.items():
                try:
                    ancestors[k] = min(ancestors[k], v)
                except KeyError:
                    ancestors[k] = v
        self.ancestors = ancestors

    def elite(self):
        self.trajectory.append('elite')

class VectorModel(Model):
    def __init__(self, config=None, ID=-1, birth_gen=-1, weights=None):
        super().__init__(config=config, ID=ID, birth_gen=birth_gen)
        if (weights is None):
            self.weights = np.random.normal(config.weight_mean,
                    config.weight_std, config.all_size)
        else:
            self.weights = weights

    def generate(self):
        self.weights = np.random.normal(self.config.weight_mean,
                self.config.weight_std, self.config.all_size)

class BitstringModel(Model):
    def __init__(self, config=None, ID=-1, birth_gen=-1, weights=None):
        super().__init__(config=config, ID=ID, birth_gen=birth_gen)
        if (weights is None):
            self.weights = [random.randint(0, 1) for _ in range(config.input_size)]
        else:
            self.weights = weights

    def generate(self):
        self.weights = [random.randint(0, 1) for _ in range(self.config.input_size)]
    
    def execute(self):
        self.bitstring = Bitstring(self.config, self.weights)
        return self.bitstring
    
    def mutate(self, coeff=1):
        self.trajectory.append('mutation')
        self.mutate_function()

    def mutate_function(self, coeff=1):
        for i in range(self.config.input_size):
            if (random.random() < 1 / self.config.input_size):
                #flip a random bit
                self.weights[i] = 1 - self.weights[i]
    
    def cross_with(self, m2, param=None):
        offspring = self.deepcopy()
        offspring.trajectory.append('crossover')
        offspring.parents = [self.ID, m2.ID]
        new_weights = self.weights.copy()

        if (self.config.cross_method == 'uniform'):
            for i in range(self.config.input_size):
                if (random.random() < 0.5):
                    new_weights[i] = m2.weights[i]
        if (self.config.cross_method == 'one_point'):
            point = random.randint(0, self.config.input_size)
            new_weights[point:] = m2.weights[point:]
        if (self.config.cross_method == 'fixed_point'):
            point = param['cross_point']
            new_weights[point:] = m2.weights[point:]
        offspring.weights = new_weights
        offspring.mutate_function()
        return offspring

    def cross_with_mask(self, m2, mask, param=None):
        offspring = self.deepcopy()
        offspring.trajectory.append('crossover')
        offspring.parents = [self.ID, m2.ID]
        new_weights = self.weights.copy()

        for j in mask:
            new_weights[j] = m2.weights[j]
        offspring.weights = new_weights
        offspring.mutate_function()
        return offspring

    def deepcopy(self):
        copy = BitstringModel(self.config)
        copy.trajectory = self.trajectory.copy()
        copy.ancestors = self.ancestors.copy()
        copy.weights = self.weights.copy()
        return copy

class FCModel(Model):
    

    def __init__(self, config=None, ID=-1, birth_gen=-1, weights=None):
        super().__init__(config=config, ID=ID, birth_gen=birth_gen)
        if (weights is None):
            self.weights = np.random.normal(config.weight_mean,
                    config.weight_std, config.all_size)
        else:
            self.weights = weights

    def generate(self):
        self.weights = np.random.normal(self.config.weight_mean,
                self.config.weight_std, self.config.all_size)

    def compile(self):
        return

    def execute(self):
        self.nn = FC(self.config, self.weights, batch_size=self.config.batch_size)
        return self.nn

    def detach(self):
        self.nn = None

    def mutate(self, coeff=1):
        self.trajectory.append('mutation')
        self.mutate_function(coeff)
        if (self.config.ea == 'speciation'):
            self.add_ancestors()

    def mutate_function(self, coeff=1):
        perturb = np.multiply(np.random.normal(0,
                self.config.perturb_std * coeff, self.config.all_size),
                np.random.choice([0,1], size=(len(self.weights)),p=[1 - self.config.weight_perturb_rate, self.config.weight_perturb_rate]))
        self.weights += perturb
        self.weights *= np.random.choice([0,1], size=len(self.weights), p = [self.config.weight_zero_rate, 1 - self.config.weight_zero_rate])

    def cross_with(self, m2, param=None):
        offspring = self.deepcopy()
        offspring.trajectory.append('crossover')
        offspring.parents = [self.ID, m2.ID]
        new_weights = self.weights.copy()

        if (self.config.cross_norm):
            new_weights *= np.linalg.norm(m2.weights) / np.linalg.norm(self.weights)

        layer_size = self.config.layer_size

        if self.config.cross_method == 'default':
            for i in range(self.config.all_size):
                number = random.random()
                if number < 0.5:
                    new_weights[i] = m2.weights[i]

        elif self.config.cross_method == 'mean':
            new_weights == (new_weights + m2.weights) / 2

        elif self.config.cross_method == 'op':
            module_num = self.config.cross_module_number
            prev = 0
            for i in range(1, len(layer_size)):
                for j in range(module_num):
                    next = prev + layer_size[i] // module_num
                    if (j % 2 == 0):
                        new_weights[prev:next] = m2.weights[prev:next]
                        prev = next

            for i in range(1, len(layer_size)):
                for j in range(module_num):
                    next = prev + layer_size[i] * layer_size[i - 1] // module_num
                    if (j % 2 == 0):
                        new_weights[prev:next] = m2.weights[prev:next]
                        prev = next

        elif self.config.cross_method == 'op2':
            module_num = self.config.cross_module_number
            prev = 0
            for i in range(1, len(layer_size)):
                for j in range(module_num):
                    next = prev + layer_size[i] // module_num
                    if (j % 2 == 0):
                        new_weights[prev:next] = m2.weights[prev:next]
                        prev = next

            for i in range(1, len(layer_size)):
                for k in range(layer_size[i - 1]):
                    for j in range(module_num):
                        next = prev + layer_size[i] // module_num
                        if (j % 2 == 0):
                            new_weights[prev:next] = m2.weights[prev:next]
                            prev = next

        elif self.config.cross_method == 'jagged':
            module_num = self.config.cross_module_number
            if module_num == 2:
                cross_mask = [0 ,1]
            else:
                mask_sum = 0
                while mask_sum != module_num / 2:
                    cross_mask = np.random.randint(2, size=module_num)
                    mask_sum = sum(cross_mask)
            for i in range(self.config.all_size):
                if cross_mask[self.config.module_identifier[i]] == 0:
                    new_weights[i] = m2.weights[i]

        elif self.config.cross_method == 'modular':
            module_num = self.config.cross_module_number
            if module_num == 2:
                groups, q = modular_cross_mask(5, layer_size, [m2.weights,new_weights], False)
                cross_mask = random.choice([[0, 1], [1, 0]])
            else:
                groups, q = modular_cross_mask(5, layer_size, [m2.weights,new_weights], False)
                module_num = len(groups)
                mask_sum = 0
                while mask_sum - module_num / 2 > 1:
                    cross_mask = np.random.randint(2, size=module_num)
                    mask_sum = sum(cross_mask)
            for j in groups[0]:
                new_weights[j] = m2.weights[j]
        elif self.config.cross_method == 'global_modular':
            module_num = self.config.cross_module_number
            if module_num == 2:
                cross_mask = random.choice([[0, 1], [1, 0]])
            else:
                module_num = len(param['global_mask'])
                mask_sum = 0
                while mask_sum - module_num / 2 > 1:
                    cross_mask = np.random.randint(2, size=module_num)
                    mask_sum = sum(cross_mask)
            for i in range(module_num):
                if cross_mask[i] == 0:
                    for j in param['global_mask'][i]:
                        new_weights[j] = m2.weights[j]


        if (self.config.ea == 'speciation'):
            offspring.add_ancestors(m2.ancestors)

        offspring.weights = new_weights
        offspring.mutate_function()
        return offspring

    def cross_with_mask(self, m2, mask, param=None):
        offspring = self.deepcopy()
        offspring.trajectory.append('crossover')
        offspring.parents = [self.ID, m2.ID]
        new_weights = self.weights.copy()

        for j in mask:
            new_weights[j] = m2.weights[j]
        if (self.config.ea == 'speciation'):
            offspring.add_ancestors(m2.ancestors)

        offspring.weights = new_weights
        offspring.mutate_function()
        return offspring

    def apply_neuron_permutation(self, permutation):
        prev = 0
        layer_size = self.config.layer_size
        new_weights = self.weights.copy()
        for i in range(1, len(layer_size) - 1):
            for j in range(layer_size[i]):
                new_weights[prev + j] = self.weights[prev + permutation[i-1][j]]
            prev += layer_size[i]

        prev += layer_size[i+1]

        for i in range(1, len(layer_size) - 1):
            self.weights = new_weights.copy()
            for j in range(layer_size[i]):
                #indices1 = np.full(layer_size[i-1], prev, dtype=int)
                #indices2 = np.full(layer_size[i-1], prev, dtype=int)
                #indices1 += [t * layer_size[i] + j for t in range(layer_size[i - 1])]
                #indices2 += [t * layer_size[i] + permutation[i-1][j] for t in range(layer_size[i - 1])]
                indices1 = [prev + t * layer_size[i] + j for t in range(layer_size[i - 1])]
                indices2 = [prev + t * layer_size[i] + permutation[i-1][j] for t in range(layer_size[i - 1])]
                new_weights[indices1] = self.weights[indices2]
                #print(indices1, indices2)
                next = prev + layer_size[i] * layer_size[i - 1]

                #indices1 = np.full(layer_size[i + 1], next + j * layer_size[i + 1], dtype=int)
                #indices2 = np.full(layer_size[i + 1], next + permutation[i-1][j] * layer_size[i + 1], dtype=int)
                #indices1 += range(layer_size[i + 1])
                #indices2 += range(layer_size[i + 1])
                indices1 = [next + j * layer_size[i + 1] + t for t in range(layer_size[i + 1])]
                indices2 = [next + permutation[i-1][j] * layer_size[i + 1] + t for t in range(layer_size[i + 1])]
                new_weights[indices1] = self.weights[indices2]

            prev = next

        self.weights = new_weights


    def deepcopy(self):
        copy = FCModel(self.config)
        copy.trajectory = self.trajectory.copy()
        copy.ancestors = self.ancestors.copy()
        copy.weights = self.weights.copy()
        return copy
