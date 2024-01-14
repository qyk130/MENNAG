import os
import pickle
from pickle import UnpicklingError

class History():

    def __init__(self):
        self.store = []
        self.summary = {}

    def __getitem__(self, key):
        return self.store[key]

    def summarize(self):
        for s in self.store:
            for k, v in s.get_summary().items():
                if k in self.summary:
                    self.summary[k].append(v)
                else:
                    self.summary[k] = [v]

    def load(self, filename):
        infile = open(filename, 'rb')
        while 1:
            try:
                item = pickle.load(infile)
                self.store.append(item)
            except (EOFError, UnpicklingError):
                break
        infile.close()
        self.summarize()
