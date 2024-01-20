import numpy as np
from configs import Configs
from nodes import Root
from model import FCModel
from crossmask import modular_cross_mask, leiden_cross_mask, tree_to_fos
from specie import Specie
from similarities import get_similarity
from ea import neuron_rearrange
import json
import pickle
import random
import pdb
import math
import time
import sys
import train
from eval.eval import eval, EvalSummary
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor, MPIPoolExecutor
genome = {
    "direct": FCModel,
    "MENNAG": Root
}

def get_FOS(parent1, parent2, FOS_type, OM_type, params):
    if FOS_type == 'LT' or OM_type == 'genepool':
        return params['FOS'].copy()
    if FOS_type == 'modularity':
        return modular_cross_mask(4, parent1.config.layer_size,
                [parent1.weights, parent2.weights], False)[0]
    if FOS_type == 'leiden':
        return leiden_cross_mask(4, parent1.config.layer_size,
                [parent1.weights, parent2.weights], False)[0]

def optimal_mixing(batch):
    def get_p(parent2, OM_type):
        if OM_type == 'recombinative':
            return parent2
        else:
            return random.choice(parent2)
    config, parents, parent_fit, ID, tasks, params = batch
    parent1, parent2 = parents
    result = None
    evals = 0
    if (config.OM_type == 'recombinative') and (config.neuron_similarity):
        neuron_rearrange((parent1, parent2, params['values1'], params['values2']))
    success = False
    if config.FOS:
        FOS = get_FOS(parent1, parent2, config.FOS_type, config.OM_type, params)
        list.sort(FOS, key=len, reverse=True)
        #print(sum([len(s) for s in FOS]))
        if config.FOS_type == 'LT':
            FOS.remove(FOS[0])
        random.shuffle(FOS)
        #print(len(FOS))
        for s in FOS:
            #print(s)
            evals += 1
            p = get_p(parent2, config.OM_type)
            offspring = parent1.cross_with_mask(p, s, params)
            offspring.ID = ID
            offspring.birth_gen = params['gen']
            offspring.FOS = s
            offspring.FOS_list = FOS
            result = eval((offspring, tasks[0], tasks[1], False, config))
            if (random.random() < config.OM_random_accept_rate) or (parent_fit < result['fit']):
                success = True
                break
    else:
        for i in range(20):
            evals += 1
            offspring = parent1.cross_with(parent2, params)
            offspring.ID = ID
            offspring.birth_gen = params['gen']
            result = eval((offspring, tasks[0], tasks[1], False, config))
            if (random.random() < config.OM_random_accept_rate) or (parent_fit < result['fit']):
                success = True
                break
    if (success):
        result['evals'] += evals - 1
        return result
    else:
        return evals

from ea import EA
class OMEA(EA):
    def __init__(self, config):
        super().__init__(config)
        self.prev_results = None

    def ask(self, task, seeds):
        self.task = task
        self.seeds = seeds
        if (len(self.pop) == 0):
            super().ask()
            self.task = task
            self.seeds = seeds
            workloads = []
            for j in range(len(self.pop)):
                workloads.append((
                        self.pop[j],
                        self.task,
                        seeds,
                        False,
                        self.config
                    ))
            results = train.perform_mpi_function(self.config.mpi, workloads, self.config.num_workers, eval)
            reducedResult = EvalSummary()
            reducedResult.reduce(results, 'pfit')
            self.results = reducedResult
            if (self.config.behavior_similarity):
                self.parent_child_behavior_sims(self.results, self.results)
                self.population_behavior_sims(self.results)

            if (self.config.phenotype_similarity):
                self.parent_child_phenotype_sims(self.results, self.results)
                self.population_phenotype_sims(self.results)
        return len(self.pop)

    def reproduce(self):
        self.gen += 1
        param = {}
        if self.config.FOS:
            self.calc_FOS()
            if self.config.FOS_type == 'LT':
                param['FOS'] = self.FOS
        #if self.prev_results is not None:
        #    self.results.store_id.update(self.prev_results.store_id)
        self.prev_results = self.results
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

        results = []
        for i in range(round(pop_size * elitism_ratio)):
            offspring = self.pop[i]
            offspring.elite()
            #offspring.compile()
            results.append(self.prev_results.store_id[offspring.ID])
        failed_evals = 0
        while (len(results) < pop_size):
            workloads = []
            while (len(workloads) < pop_size - len(results)):
                parent1 = np.random.choice(self.pop, p=p)
                parent2 = random.choice(self.pop)
                if (parent1 != parent2):
                    params = {}
                    params.update(param)
                    if (self.config.neuron_similarity):
                        params['values1'] = self.prev_results.get_res_by_ID(parent1.ID, 'neuron_values')
                        params['values2'] = self.prev_results.get_res_by_ID(parent2.ID, 'neuron_values')
                    parent_fit = self.prev_results.get_res_by_ID(parent1.ID, 'fit')
                    if self.config.OM_type == 'genepool':
                        parent2 = self.pop
                    elif self.config.cross_method == 'no_cross':
                        parent2 = parent1
                    workloads.append((self.config, (parent1, parent2), parent_fit, self.next_id(), (self.task, self.seeds), params))
            tmp_result = (train.perform_mpi_function(self.config.mpi, workloads, self.config.num_workers, optimal_mixing))
            for r in tmp_result:
                if (isinstance(r, int)):
                    failed_evals += r
                else:
                    results.append(r)
        reduced_result = EvalSummary()
        reduced_result.reduce(results, 'pfit')
        self.results = reduced_result
        new_pop = self.results.get_res('pop')
        #print(self.results.evals)
        self.results.evals += failed_evals
        #print(self.results.evals)
        if (self.config.behavior_similarity):
            self.parent_child_behavior_sims(self.results, self.prev_results)
            self.population_behavior_sims(self.results)

        if (self.config.phenotype_similarity):
            self.parent_child_phenotype_sims(self.results, self.prev_results)
            self.population_phenotype_sims(self.results)
        self.pop = new_pop
        self.prev_results.summarize()

        return self.results, self.results.evals

    def end(self):
        self.prev_results = self.results
        self.prev_results.summarize()
        if (self.config.behavior_similarity):
            self.parent_child_behavior_sims(self.results, self.prev_results)
            self.population_behavior_sims(self.results)

        if (self.config.phenotype_similarity):
            self.parent_child_phenotype_sims(self.results, self.prev_results)
            self.population_phenotype_sims(self.results)

    def calc_FOS(self):

        def cluster_size(c):
            if (len(c) == 1):
                return 1
            else:
                return cluster_size(c[0]) + cluster_size(c[1])

        def tree_to_list(tree, l):
            if isinstance(tree, list):
                for node in tree:
                    tree_to_list(node, l)
            else:
                l.append(tree)
            return l

        def MI(Xs, Ys, cov, std, store):
            if (Xs, Ys) in store:
                return store[(Xs, Ys)]
            X = list(Xs)
            Y = list(Ys)
            if (len(X) == 1) and (len(Y) == 1):
                tmp = 1 / (1 - (cov[X[0], Y[0]] / (std[X[0]] * std[Y[0]])) ** 2)
                if tmp < 0:
                    print(tmp, X, Y, cov[X[0], Y[0]], std[X[0]], std[Y[0]])
                mi = np.log(np.sqrt(tmp))
                store[(Xs, Ys)] = mi
                return mi
            if (len(X) == 1):
                #mi = cluster_size(Y[0]) / cluster_size(Y) * MI(X, Y[0], cov, std) \
                #        + cluster_size(Y[1]) / cluster_size(Y) * MI(X, Y[1], cov, std)
                mi = 1 / len(Y) * MI(frozenset(X), frozenset([Y[0]]), cov, std, store) \
                        + (len(Y) - 1) / len(Y) * MI(frozenset(X), frozenset(Y[1:]), cov, std, store)
            else:
                #mi = cluster_size(X[0]) / cluster_size(X) * MI(X[0], Y, cov, std) \
                #        + cluster_size(X[1]) / cluster_size(X) * MI(X[1], Y, cov, std)
                mi = 1 / len(X) * MI(frozenset([X[0]]), Ys, cov, std, store) \
                        + (len(X) - 1) / len(X) * MI(frozenset(X[1:]), Ys, cov, std, store)
            store[(Xs, Ys)] = mi
            return mi

        if self.config.FOS_type == 'LT':
            pop_weights = np.zeros((len(self.pop[0].weights), len(self.pop)))
            for i in range(len(self.pop)):
                pop_weights[:,i] = self.pop[i].weights
            cov = np.cov(pop_weights)
            std = np.sqrt(cov.diagonal())
            clusters = [[i] for i in range(len(self.pop[0].weights))]
            store = {}
            for i in range(len(self.pop[0].weights) - 2):
                l = len(clusters)
                max = -1000000
                max_pair = None
                for j in range(l - 1):
                    for k in range(j + 1, l):
                        X = []
                        X = frozenset(tree_to_list(clusters[j], X))
                        #print(X, clusters[j])
                        Y = []
                        Y = frozenset(tree_to_list(clusters[k], Y))
                        similarity = MI(X, Y, cov, std, store)
                        if similarity > max:
                            max = similarity
                            max_pair = (j, k)
                merged = [clusters[max_pair[0]], clusters[max_pair[1]]]
                clusters.remove(merged[0])
                clusters.remove(merged[1])
                clusters.append(merged)
            self.FOS = tree_to_fos(clusters, 4, 0)

    def write_history(self, filename):
        if (not self.writeInit):
            outfile = open(filename, 'wb+')
            self.writeInit = True
        else:
            outfile = open(filename, 'ab')
        tmp = self.prev_results.store_id
        self.prev_results.store_id = None
        #print(self.prev_results.get_res("parent_child_behavior_sims"))
        pickle.dump(self.prev_results, outfile)
        self.prev_results.store_id = tmp
        outfile.close()
