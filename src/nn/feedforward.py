import numpy as np
from connlist import ConnList
from activations import get_act
import pdb


class FeedForward():

    def __init__(self, configs):
        input_size = configs.input_size
        output_size = configs.output_size
        self.compiled = False
        self.nodeCount = input_size + output_size
        self.nodeLookup = {}
        self.connList = ConnList()
        self.inputs = np.zeros((input_size))
        self.outputs = np.zeros((input_size))
        self.input_size = input_size
        self.output_size = output_size
        self.values = None
        self.biases = []
        self.acts = []
        self.rawConnList = []
        self.penalty = 1
        for i in range(self.input_size):
            self.acts.append(get_act(0))
            self.biases.append(0)
        for i in range(self.output_size):
            self.acts.append(get_act(1))
            self.biases.append(0)

    def get_penalty(self):
        return self.penalty

    def add_node(self, nodeName, node):
        try:
            if (nodeName[0].isdigit()):
                if (not (nodeName in self.nodeLookup)):
                    self.nodeLookup[nodeName] = self.nodeCount
                    self.biases.append(node[0])
                    self.acts.append(get_act(node[1]))
                    self.nodeCount += 1
        except IndexError:
            print("indexerror in add_node ", nodeName, node)

    def add_conn(self, source, target, weight):
        try:
            if (source[0] == 'i'):
                sourceId = int(source[1:])
            else:
                sourceId = self.nodeLookup[source]
            if (target[0] == 'o'):
                targetId = int(target[1:]) + self.input_size
            else:
                targetId = self.nodeLookup[target]
            self.connList.add_conn(sourceId, targetId, weight)
            self.rawConnList.append((source, target, weight))
        except IndexError:
            print("indexerror in add_conn ", source, terget)

    def topological_sort(self, node, nodeFlags, queue):
        if (nodeFlags[node] != 0):
            return
        indices = self.connList.get_all_conn_indices_source(node)
        if (indices is not None):
            targets = self.connList.connTarget[indices]
            nodeFlags[node] = 1
            for target in targets:
                self.topological_sort(target, nodeFlags, queue)
        nodeFlags[node] = 2
        queue.append(node)

    def traverse_from_output(self, node, nodeFlags):
        indices = self.connList.get_all_conn_indices_target(node)
        nodeFlags[node] = 1
        if (indices is not None):
            toList = self.connList.connSource[indices]
            for to in toList:
                if (nodeFlags[to] == 0):
                    self.traverse_from_output(to, nodeFlags)

    def traverse_from_input(self, node, nodeFlags):
        indices = self.connList.get_all_conn_indices_source(node)
        nodeFlags[node] = 1
        if (indices is not None):
            toList = self.connList.connTarget[indices]
            for to in toList:
                if (nodeFlags[to] == 0):
                    self.traverse_from_input(to, nodeFlags)

    def remove_redundent(self):
        original_size = self.connList.connCount
        self.connList.sort_by_target()
        flags1 = np.zeros(self.nodeCount, dtype=int)
        flags1[0:self.input_size + self.output_size] = 1
        for i in range(self.input_size, self.input_size + self.output_size):
            self.traverse_from_output(i, flags1)
        flags2 = np.zeros(self.nodeCount, dtype=int)
        flags2[0:self.input_size + self.output_size] = 1
        self.connList.sort_by_source()
        for i in range(self.input_size):
            self.traverse_from_input(i, flags2)
        flags = flags1 & flags2
        #flags = flags1
        #print(flags1,flags2,flags)
        self.nodeCount = sum(flags)
        newActs = []
        newBiases = []
        newIndices = np.zeros(len(flags), dtype=int)
        c = 0
        for i in range(len(flags)):
            if (flags[i] == 1):
                newIndices[i] = c
                newActs.append(self.acts[i])
                newBiases.append(self.biases[i])
                c += 1
            else:
                newIndices[i] = -1
        self.acts = newActs
        self.biases = newBiases
        self.connList.remove_redundent(newIndices)
        if original_size > 0:
            self.penalty = 1 - self.connList.connCount / original_size
        else:
            self.penalty = 1

    def add_finish(self):
        self.biases = np.array(self.biases, dtype=np.float16)
        self.connList.compile()
        try:
            self.remove_redundent()
        except IndexError:
            print('remove redundent index error')
            exit()
        self.nodeLookup = None


    def compile(self):
        try:
            self.connList.sort_by_source()
            nodeFlags = np.zeros(self.nodeCount, dtype=int)
            topologicalOrder = []
            for node in range(self.nodeCount):
                self.topological_sort(node, nodeFlags, topologicalOrder)
            topologicalOrder.reverse()
            self.connList.sort_by_target()
            sortIndex = []
            for node in topologicalOrder:
                indices = self.connList.get_all_conn_indices_target(node)
                if (indices is not None):
                    sortIndex.extend(indices)
            self.connList.apply_indices(sortIndex)
            if (self.connList.connCount != len(sortIndex)):
                self.connList.connCount = len(sortIndex)
            self.values = np.zeros((self.nodeCount))
        except IndexError:
            print('compile index error')
            exit()

    def step(self, inputs):
        self.values.fill(0)
        self.values[0:self.input_size] = inputs
        #print(len(self.values) - len(self.biases))
        self.values[0:len(self.values)] += self.biases
        targetList = self.connList.connTarget
        sourceList = self.connList.connSource
        weightList = self.connList.connWeight
        connCount = self.connList.connCount
        if (connCount == 0):
            return self.values[self.input_size:self.input_size + self.output_size]
        prev = targetList[0]
        i = 0
        try:
            while (i < connCount):
                while ((i < connCount) & (prev == targetList[i])):
                    self.values[targetList[i]] += self.values[sourceList[i]] * \
                        weightList[i]
                    i += 1
                    if (i >= connCount):
                        break
                self.values[prev] = self.acts[prev](self.values[prev])
                #pdb.set_trace()
                if (i >= connCount):
                    break
                prev = targetList[i]
        except IndexError:
            print('step index error', i, connCount, len(targetList), len(self.rawConnList))
            exit()
        output = self.values[self.input_size:self.input_size + self.output_size]
        return output.copy()
