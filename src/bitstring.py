import numpy as np

class Bitstring():
    def __init__(self, size, weights=None):
        self.size = size
        if weights == None:
            self.bits = [0] * size
        else: 
            self.bits = weights
            
    
    def step(self, input):
        return self.bits, {}
    
    def compile(self):
        return
    
    def get_penalty(self):
        return 0