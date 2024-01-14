class Specie():

    def __init__(self):
        self.pop = []
        self.best_fit = -10000.0
        self.avg_fit = 0.0
        self.leader = None
        self.space = 0

    def add(self, ind, fit):
        self.pop.append(ind)
        if (fit > self.best_fit):
            self.best_fit = fit
            self.leader = ind
        self.avg_fit = self.avg_fit * (len(self.pop) - 1) / len(self.pop)
