import numpy as np
from activations import *

class FC():
    def __init__(self, config, all, batch_size=1):
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.layer_size = config.layer_size
        self.all_size = config.all_size
        self.act = np.tanh
        self.output_act = np.tanh
        self.values = []
        self.weights = []
        self.biases = []
        self.config = config
        self.batch_size = batch_size
        prev = 0
        for i in range(1, len(self.layer_size)):
            self.biases.append(all[prev:prev + self.layer_size[i]])
            prev += self.layer_size[i]

        for i in range(1, len(self.layer_size)):
            self.weights.append(all[prev:prev +
                    self.layer_size[i - 1] * self.layer_size[i]])
            self.weights[i - 1] = self.weights[i - 1].reshape(
                    (self.layer_size[i - 1],self.layer_size[i]))
            prev += self.layer_size[i] * self.layer_size[i - 1]

        for b in self.biases:
            self.values.append(np.zeros((batch_size, b.shape[0])))

    def step(self, inputs):
        stats = {}
        #if len(inputs.shape) == 1:
        #    inputs = inputs.reshape((self.batch_size, inputs.shape[0]))
        for i in range(len(self.values)):
            self.values[i].fill(0)
            self.values[i] += self.biases[i]
            wt = self.weights[0].T
            it = inputs.T
            dotted = np.dot(wt, it)
            try:
                self.values[0] += dotted.T
            except TypeError:
                print(self.values, dotted)
        self.values[0] = self.act(self.values[0])

        for i in range(1, len(self.layer_size) - 2):
            try:
                wt = self.weights[i].T
                it = self.values[i - 1].T
            except TypeError:
                print(wt, it)
            dotted = np.dot(wt, it)
            self.values[i] += dotted.T
            self.values[i] = self.act(self.values[i])
        i = len(self.layer_size) - 2
        wt = self.weights[i].T
        it = self.values[i - 1].T
        try:
            dotted = np.dot(wt, it)
        except ValueError:
            print(wt,it)
        try:
            self.values[i] += dotted.T 
        except ValueError:
            print(self.values, self.weights)
        self.values[i] = self.act(self.values[i])
        if self.config.neuron_similarity:
            stats['neuron_values'] = self.values[0:-1]
        return self.values[-1], stats

    def get_penalty(self):
        penalty = 0
        if (self.config.l1):
            for weight in self.weights:
                penalty += sum(sum(abs(weight))) * self.config.l1_coef
        if (self.config.l0):
            for weight in self.weights:
                penalty += self.config.l0_coef * sum(sum(abs(weight) < self.config.l0_threshold))
        #print(penalty)
        return penalty

    def compile(self):
        return
