import os

class History():

    def __init__(self, taskNum):
        self.tasks = []
        self.seeds = []
        self.bests = []
        self.fitnesses = []
        self.bestFit = []

    def append(self, task, seed, best, fitnesses):
        self.tasks.append(task)
        self.seeds.append(seed)
        self.fitnesses.append(fitnesses)
        self.bestFit.append(fitnesses[0])
        self.bests.append(best)
