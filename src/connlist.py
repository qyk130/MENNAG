import numpy as np


class ConnList():

    def __init__(self):
        self.connDict = {}
        self.connCount = 0

    def add_conn(self, source, target, weight):
        if (source, target) in self.connDict:
            self.connDict[(source, target)] += weight
        else:
            self.connDict[(source, target)] = weight

    def compile(self):
        try:
            conns = [(k[0], k[1], v) for k, v in self.connDict.items()]
            conns = list(zip(*conns))
            self.connSource = np.asarray(conns[0], dtype=np.int)
            self.connTarget = np.asarray(conns[1], dtype=np.int)
            self.connWeight = np.asarray(conns[2])
            self.connCount = len(self.connSource)
            self.connDict = None
        except IndexError:
            self.connSource = np.asarray([], dtype=np.int)
            self.connTarget = np.asarray([], dtype=np.int)
            self.connWeight = np.asarray([])
            self.connCount = len(self.connSource)
            self.connDict = None

    def sort_by_source(self):
        sortedIndex = np.argsort(self.connSource)
        self.apply_indices(sortedIndex)

    def sort_by_target(self):
        sortedIndex = np.argsort(self.connTarget)
        self.apply_indices(sortedIndex)

    def get_all_conn_indices_source(self, node):
        left = np.searchsorted(self.connSource, node)
        try:
            if (left == self.connCount or left < 0):
                return None
            if (self.connSource[left] != node):
                return None
            right = left
            while ((self.connSource[left - 1] == node) and (left != 0)):
                left -= 1
            while (self.connSource[right] == node):
                if (right + 1 == self.connCount):
                    right = self.connCount
                    break
                right += 1
        except IndexError:
            print(self.connSource)

        return np.arange(left, right)

    def get_all_conn_indices_target(self, node):
        left = np.searchsorted(self.connTarget, node)
        try:
            if (left == self.connCount or left < 0):
                return None
            if (self.connTarget[left] != node):
                return None
            right = left
            while ((self.connTarget[left - 1] == node) and (left != 0)):
                left -= 1
            while (self.connTarget[right] == node):
                if (right + 1 == self.connCount):
                    right = self.connCount
                    break
                right += 1
        except IndexError:
            print(self.connSource)
        return np.arange(left, right)

    def remove_redundent(self, indices):
        newSources = []
        newTargets = []
        newWeights = []
        for i in range(self.connCount):
            s = indices[self.connSource[i]]
            t = indices[self.connTarget[i]]
            if ((s != -1) & (t != -1)):
                newSources.append(s)
                newTargets.append(t)
                newWeights.append(self.connWeight[i])
        self.connSource = np.asarray(newSources, dtype=np.int)
        self.connTarget = np.asarray(newTargets, dtype=np.int)
        self.connWeight = np.asarray(newWeights)
        self.connCount = len(self.connSource)

    def apply_indices(self, sortedIndex):
        self.connSource = self.connSource[sortedIndex]
        self.connTarget = self.connTarget[sortedIndex]
        self.connWeight = self.connWeight[sortedIndex]
