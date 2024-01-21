import os
import numpy as np
from configs import Configs
from nodes import Root
from model import FCModel, BitstringModel, min_common_ancestry_dist
from crossmask import modular_cross_mask
from specie import Specie
from similarities import get_similarity
import json
import pickle
import random
import pdb
import math
import time
import sys
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor, MPIPoolExecutor
import train
genome = {
    "direct": FCModel,
    "MENNAG": Root,
    "bitstring": BitstringModel
}

def crossover(batch):
    config, parents, ID, params = batch
    parent1, parent2 = parents
    if config.neuron_similarity:
        neuron_rearrange((parent1, parent2, params['values1'], params['values2']))
    offspring = parent1.cross_with(parent2, params)
    offspring.ID = ID
    offspring.birth_gen = params['gen']
    return offspring

def neuron_rearrange(batch):
    ind1, ind2, best_values, values= batch
    permutation = []
    for j in range(len(values[0])):
        perm = []
        candidate = list(range(values[0][j].shape[1]))
        for l in range(values[0][j].shape[1]):
            diff = np.zeros(values[0][j].shape[1])
            for k in range(len(values)):
                for m in range(values[k][j].shape[0]):
                    diff += np.abs(values[k][j][m] - best_values[k][j][m,l])
            order = np.argsort(diff)
            for c in order:
                if c in candidate:
                    perm.append(c)
                    candidate.remove(c)
                    break
        permutation.append(perm)
    ind2.apply_neuron_permutation(permutation)
    return ind2

class EA():

    def __init__(self, config):
        self.pop = []
        self.config = config
        self.writeInit = False
        self.gen = 0
        self.counter = -1
        self.results = None

    def ask(self):
        if (len(self.pop) == 0):
            for i in range(self.config.pop_size):
                newInd = genome[self.config.encoding](
                        config=self.config,
                        ID=self.next_id(),
                        birth_gen=0)
                newInd.generate()
                #newInd.compile()
                self.pop.append(newInd)
            return self.pop
        else:
            #self.reproduce()
            return self.pop

    def tell(self, results, task, seed):
        self.pop = results.get_res('pop')
        self.performance = results.get_res('fit')
        self.fitnesses = results.get_metric()

        if (self.config.behavior_similarity):
            if (self.config.cross_rate > 0):
                self.parent_child_behavior_sims(results, self.results)
            self.population_behavior_sims(results)

        if (self.config.phenotype_similarity):
            if (self.config.cross_rate > 0):
                self.parent_child_phenotype_sims(results, self.results)
            self.population_phenotype_sims(results)
        '''
        if (self.config.neuron_similarity):
            if self.config.mpi:
                workloads = []
                count = 0
                for ind in self.pop:
                    workloads.append((
                        self.pop[0],
                        ind,
                        results.get_res_by_ID(self.pop[0].ID, 'neuron_values'),
                        results.get_res_by_ID(ind.ID, 'neuron_values'),
                        count))
                    count += 1
                self.pop = train.perform_mpi_function(workloads, self.config.num_workers, neuron_rearrange)
            else:
                for ind in self.pop:
                    neuron_rearrange((
                        self.pop[0],
                        ind,
                        results.get_res_by_ID(self.pop[0].ID, 'neuron_values'),
                        results.get_res_by_ID(ind.ID, 'neuron_values')))
                        '''
        self.gen += 1
        #results.summarize()
        self.results = results

    def rank(self):
        rank = np.argsort(self.fitnesses)
        return (rank + 1) / sum(rank)

    def next_id(self):
        self.counter += 1
        return self.counter

    def reproduce(self):
        param = {}
        param['gen'] = self.gen
        if (self.config.cross_method == 'global_modular'):
            all_weights= []
            for p in self.pop:
                all_weights.append(p.weights)
            global_mask = modular_cross_mask(self.config.layer_size, all_weights, False)
            param['global_mask'] = global_mask

        new_pop = []
        pop_size = self.config.pop_size
        elitism_ratio = self.config.elitism_ratio
        p = np.flip(np.array(range(self.config.pop_size))) + 1
        p = p / sum(p)

        workloads = []

        for i in range(round(pop_size * elitism_ratio)):
            offspring = self.pop[i].deepcopy()
            offspring.ID = self.pop[i].ID
            offspring.elite()
            #offspring.compile()
            new_pop.append(offspring)

        while (len(workloads) + len(new_pop) < round(pop_size * self.config.cross_rate)):
            parent1 = np.random.choice(self.pop, p=p)
            parent2 = random.choice(self.pop)
            if (parent1 != parent2):
                params = {}
                params.update(param)
                if (self.config.neuron_similarity):
                    params['values1'] = self.results.get_res_by_ID(parent1.ID, 'neuron_values')
                    params['values2'] = self.results.get_res_by_ID(parent2.ID, 'neuron_values')
                workloads.append((self.config, (parent1, parent2), self.next_id(), params))
        offsprings = train.perform_mpi_function(self.config.mpi, workloads, self.config.num_workers, crossover)
        new_pop.extend(offsprings)

        
        while(len(new_pop) < pop_size):
            try:
                offspring = np.random.choice(self.pop, p=p).deepcopy()
            except ValueError:
                print(len(self.pop),len(p))
            offspring.ID = self.next_id()
            offspring.mutate()
            offspring.birth_gen = self.gen
            new_pop.append(offspring)
        self.pop = new_pop
        self.results.summarize()

    def stop(self):
        print(self.config.generation)
        if (self.gen >= self.config.generation):
            return True
        return False


    def population_phenotype_sims(self, results):
        pop_sims = []
        while len(pop_sims) < 1000:
            p1 = random.choice(list(results.store_id.keys()))
            p2 = random.choice(list(results.store_id.keys()))
            p1 = results.get_res_by_ID(p1, "pop")
            p2 = results.get_res_by_ID(p2, "pop")
            if (p1 is not p2):
                pop_sims.append(get_similarity(self.config.phenotype_similarity_measure,
                        p1.weights, p2.weights))
        results.add_summary_key("population_phenotype_sims")
        results.set_summary("population_phenotype_sims", np.mean(pop_sims))

    def parent_child_phenotype_sims(self, current_results, prev_results):
        if self.gen == 0:
            current_results.add_summary_key("parent_child_phenotype_sims")
            current_results.add_summary_key("parent_phenotype_sims")
            current_results.set_summary("parent_child_phenotype_sims", 1.0)
            current_results.set_summary("parent_phenotype_sims", 1.0)
        else:
            parent_child_sims = []
            parent_sims = []
            id_list = list(current_results.store_id.keys())
            for id in id_list:
                p = current_results.get_res_by_ID(id, "pop")
                if ((p.birth_gen == self.gen) and (p.trajectory[-1] == 'crossover')):
                    parents = p.parents
                    p0 = current_results.get_res_by_ID(p.ID, "pop")
                    p1 = prev_results.get_res_by_ID(parents[0], "pop")
                    p2 = prev_results.get_res_by_ID(parents[1], "pop")
                    s1 = get_similarity(self.config.phenotype_similarity_measure,
                            p0.weights, p1.weights)
                    s2 = get_similarity(self.config.phenotype_similarity_measure,
                            p0.weights, p2.weights)
                    parent_sims.append(get_similarity(self.config.phenotype_similarity_measure,
                            p1.weights, p2.weights))
                    parent_child_sims.append((s1 + s2) / 2)
            current_results.add_summary_key("parent_child_phenotype_sims")
            current_results.add_summary_key("parent_phenotype_sims")
            if (len(parent_sims) > 0):
                current_results.set_summary("parent_phenotype_sims", np.mean(parent_sims))
            else:
                current_results.set_summary("parent_phenotype_sims", 1.0)
            if (len(parent_child_sims) > 0):
                current_results.set_summary("parent_child_phenotype_sims", np.mean(parent_child_sims))
            else:
                current_results.set_summary("parent_child_phenotype_sims", 1.0)

    def population_behavior_sims(self, results):
        pop_sims = []
        while len(pop_sims) < 1000:
            p1 = random.choice(list(results.store_id.keys()))
            p2 = random.choice(list(results.store_id.keys()))
            p1 = results.get_res_by_ID(p1, "pop")
            p2 = results.get_res_by_ID(p2, "pop")
            if (p1 is not p2):
                a1 = results.get_res_by_ID(p1.ID, "actions")
                a2 = results.get_res_by_ID(p2.ID, "actions")
                pop_sims.append(get_similarity(self.config.behavior_similarity_measure, a1, a2))
        results.add_summary_key("population_behavior_sims")
        results.set_summary("population_behavior_sims", np.mean(pop_sims))

    def parent_child_behavior_sims(self, current_results, prev_results):
        if self.gen == 0:
            current_results.add_summary_key("parent_child_behavior_sims")
            current_results.add_summary_key("parent_behavior_sims")
            current_results.set_summary("parent_child_behavior_sims", 1.0)
            current_results.set_summary("parent_behavior_sims", 1.0)
        else:
            parent_child_sims = []
            parent_sims = []
            id_list = list(current_results.store_id.keys())
            for id in id_list:
                p = current_results.get_res_by_ID(id, "pop")
                if ((p.birth_gen == self.gen) and (p.trajectory[-1] == 'crossover')):
                    parents = p.parents
                    a0 = current_results.get_res_by_ID(p.ID, "actions")
                    a1 = prev_results.get_res_by_ID(parents[0], "actions")
                    a2 = prev_results.get_res_by_ID(parents[1], "actions")
                    s1 = get_similarity(self.config.behavior_similarity_measure, a0, a1)
                    s2 = get_similarity(self.config.behavior_similarity_measure, a0, a2)
                    parent_sims.append(get_similarity(self.config.behavior_similarity_measure, a1, a2))
                    parent_child_sims.append((s1 + s2) / 2)
            current_results.add_summary_key("parent_child_behavior_sims")
            current_results.add_summary_key("parent_behavior_sims")
            if (len(parent_sims) > 0):
                current_results.set_summary("parent_behavior_sims", np.mean(parent_sims))
            else:
                current_results.set_summary("parent_behavior_sims", 1.0)
            if (len(parent_child_sims) > 0):
                current_results.set_summary("parent_child_behavior_sims", np.mean(parent_child_sims))
            else:
                current_results.set_summary("parent_child_behavior_sims", 1.0)

    def write_history(self, filename):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if (not self.writeInit):
            outfile = open(filename, 'wb')
            self.writeInit = True
        else:
            outfile = open(filename, 'ab')
        tmp = self.results.store_id
        self.results.store_id = None
        pickle.dump(self.results, outfile)
        self.results.store_id = tmp
        outfile.close()

def get_valid_parents(parent1, pop, range):
    parents = []
    for parent2 in pop:
        dist = min_common_ancestry_dist(parent1, parent2)
        if ((dist >= range[0]) and (dist <= range[1])):
            parents.append(parent2)
    return parents

class EASpeciation(EA):

    def __init__(self, config):
        super().__init__(config)

    def speciate(self):
        species = []
        pop = self.pop
        speciated = [False] * len(pop)
        count = 0
        while (len(pop) > count and len(species) < 50):
            specie = Specie()
            for i in range(len(pop)):
                if (not speciated[i]):
                    if (len(specie.pop) == 0):
                        specie.add(pop[i], self.fitnesses[i])
                        speciated[i] = True
                        count += 1
                    else:
                        if (min_common_ancestry_dist(specie.leader, pop[i])
                                <= self.config.speciation_range):
                            specie.add(pop[i], self.fitnesses[i])
                            speciated[i] = True
                            count += 1
            species.append(specie)

        self.species = species
        #for s in species:
        #    if (len(s.pop) > 1):
        #        self.species.append(s)
        species_len = []
        for s in self.species:
            species_len.append(len(s.pop))

        #print(pop[0].ancestors)

    def reproduce(self):
        if (self.gen <= self.config.crossover_range[0]):
            super().reproduce()
        else:
            self.speciate()
            #print("species: ", len(self.species), "population: ", len(self.pop))
            new_pop = []
            species_fit = np.array([s.best_fit for s in self.species])
            print(species_fit)
            species_space = np.flip(np.array(range(len(self.species)))) + 1
            species_space = species_space / sum(species_space)
            species_space *= self.config.pop_size
            for s in range(len(self.species)):
                specie = self.species[s]
                offsprings = []
                if (self.config.fitness_sharing):
                    p = np.ones(len(specie.pop))
                    p = p / sum(p)
                else:
                    p = np.flip(np.array(range(len(specie.pop)))) + 1
                    p = p / sum(p)
                while (len(offsprings) < round(species_space[s] * self.config.cross_rate)):
                    parent1 = np.random.choice(specie.pop, p=p)
                    if (random.random() < self.config.random_crossover_rate):
                        parent2 = random.choice(self.pop)
                    else:
                        parents = get_valid_parents(parent1, specie.pop,
                                self.config.crossover_range)
                        if (len(parents) == 0):
                            parent2 = random.choice(self.pop)
                        else:
                            parent2 = random.choice(parents)
                    if (parent1 != parent2):
                        offspring = parent1.cross_with(parent2)
                        offspring.ID = self.next_id()
                        offspring.birth_gen = self.gen
                        #offspring.compile()
                        offsprings.append(offspring)
                if (s < (int(np.ceil(len(self.pop) * self.config.elitism_ratio)))):
                    offspring = specie.pop[0].deepcopy()
                    offspring.elite()
                    offsprings.append(offspring)
                while (len(offsprings) < round(species_space[s])):
                    offspring = np.random.choice(specie.pop, p=p).deepcopy()
                    offspring.ID = self.next_id()
                    offspring.birth_gen = self.gen
                    offspring.mutate()
                    offsprings.append(offspring)
                new_pop.extend(offsprings)
            self.pop = new_pop
            #pdb.set_trace()

class EAMapelite(EA):

    def __init__(self, config):
        self.pop = []
        self.pop_lookup = {}
        self.config = config
        self.writeInit = False
        self.gen = 0
        self.counter = 0
        self.grid = []
        self.resolution = [1, 1]

    def get_grid_indices(self, features):
        x = math.floor(features[0] / ((self.config.feature_range[0][1] -
            self.config.feature_range[0][0]) / self.resolution[0]))
        x = min(self.resolution[0], x)
        y = math.floor(features[1] / ((self.config.feature_range[1][1] -
            self.config.feature_range[1][0]) / self.resolution[1]))
        y = min(self.resolution[1], y)
        return x, y

    def add_grid(self, ind, fitness, features):
        x, y = self.get_grid_indices(features)
        block = self.grid[x][y]
        if block is None:
            self.pop_lookup[features] = ind
            self.grid[x][y] = (fitness, features)
        elif (fitness > block[0]):
            del self.pop_lookup[block[1]]
            self.pop_lookup[features] = ind
            self.grid[x][y] = (fitness, features)

    def ask(self):
        if (len(self.pop) == 0):
            for i in range(self.config.initial_batch):
                newInd = Root(config=self.config, ID=self.next_id())
                newInd.generate()
                #newInd.compile()
                self.pop.append(newInd)
            return self.pop
        else:
            self.reproduce()
            return self.pop

    def tell(self, results, task, seed):
        self.results = results
        self.pop = results.get_res('pop')
        self.fitnesses = results.get_metric()
        self.features = results.get_res('me_features')
        results.summarize()

        for row in self.config.resolution:
            if self.gen == row[0]:
                self.resolution = row[1]
                old_grid = self.grid
                self.grid = [[None for i in range(self.resolution[0])]
                        for j in range(self.resolution[1])]
                for row in old_grid:
                    for block in row:
                        if (block is not None):
                            self.add_grid(self.pop_lookup[block[1]], block[0], block[1])
                break
        self.gen += 1

        for i in range(len(self.pop)):
            self.add_grid(self.pop[i], self.fitnesses[i], self.features[i])

    def reproduce(self):
        self.pop = []
        all = list(self.pop_lookup.values())
        while (len(self.pop) < self.config.batch_size):
            offspring = random.choice(all).deepcopy()
            offspring.ID = self.next_id()
            offspring.mutate()
            self.pop.append(offspring)

    def write_history(self, filename):
        fitnesses = []
        features = []
        count = 0
        for row in self.grid:
            for block in row:
                if block is not None:
                    count += 1
                    fitnesses.append(block[0])
                    features.append(block[1])
        #print(count, len(self.pop_lookup))
        indices = np.argsort(fitnesses)
        top = []
        for i in range(math.ceil(len(indices))):
            top.append((features[indices[i]], fitnesses[indices[i]],
                    self.pop_lookup[features[indices[i]]]))
        outfile = open(filename, 'wb')
        pickle.dump((self.grid, top), outfile)
        outfile.close()

class EACMA(EA):

    def __init__(self, config):
        import cma
        super().__init__(config)
        self.es = cma.CMAEvolutionStrategy(config.all_size * [0.0], 0.5,{
                'maxfevals': config.max_evals,
                'maxiter': 100000,
                'tolstagnation': 1000000,
                'tolflatfitness': 1000000,
                'CMA_elitist': False,
                "popsize": self.config.pop_size
        })

    def ask(self):
        self.pop = []
        solutions = self.es.ask()
        for s in solutions:
            ind = FCModel(self.config, self.next_id(), weights=s)
            self.pop.append(ind)
        return self.pop

    def tell(self, results, task, seed):
        import cma
        self.results = results
        if (self.config.behavior_similarity):
            #self.parent_child_behavior_sims(results)
            self.population_behavior_sims(results)

        if (self.config.phenotype_similarity):
            #self.parent_child_phenotype_sims(results)
            self.population_phenotype_sims(results)
        self.pop = results.get_res('pop')
        self.performance = results.get_res('fit')
        self.fitnesses = results.get_metric()
        results.summarize()
        self.gen += 1
        solutions = [p.weights for p in self.pop]
        f = [-fit for fit in self.fitnesses]
        self.es.tell(solutions, f)

    def stop(self):
        return self.es.stop()
