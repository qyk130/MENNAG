import matplotlib.pyplot as plt
import numpy as np
import os
from linkage import Linkage
from mpi4py.futures import MPICommExecutor, MPIPoolExecutor
import random

#mpirun --oversubscribe -n 4 python -m mpi4py.futures linkage_test.py

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def perform_mpi_function(workloads, num_workers, func):
    success = False
    result_list = []
    if True:
        for w in workload:
            result_list.append(func(w))
    else:
        while (success is False):
            try:
                with MPIPoolExecutor(max_workers = num_workers) as executor:
                    if executor is not None:
                        results = executor.map(func, workloads)
                        success = True
            except OverflowError:
                success = False
        for row in results:
            result_list.append(row)
    return result_list

def royal_road(bitstring):
    return np.sum([bitstring[i] * bitstring[i+1] * bitstring[i+2] * bitstring[i+3] * bitstring[i+4] * 5 for i in range(0, len(bitstring), 5)]), {}

def addand(bitstring):
    """ Calculate the fitness of a bitstring. Fitness is the sum of XOR of pairs of bits. """
    return np.sum([bitstring[i] * bitstring[i+1] * 2 for i in range(0, len(bitstring), 2)])

def nk_trap(bitstring):
    k = 3
    f = 0
    for i in range(0, len(bitstring), k):
        ones = sum(bitstring[i:i + k])
        if (ones == k):
            f += k
        else:
            f += k - ones - 1
    return f

def onemax(bitstring):
    return np.sum(bitstring)

def special_crossover(parent1, parent2):
    """ Perform special crossover where pairs of bits are treated as a unit. """
    block_size = 4
    offspring = np.empty_like(parent1)
    for i in range(0, len(parent1), block_size):
        if np.random.rand() < 0.5:
            offspring[i:i+block_size] = parent1[i:i+block_size]
        else:
            offspring[i:i+block_size] = parent2[i:i+block_size]
    return offspring

def uniform_crossover(parent1, parent2):
    """ Perform uniform crossover. """
    offspring = np.array([parent1[i] if np.random.rand() < 0.5 else parent2[i] for i in range(len(parent1))])
    return offspring

def onepoint_crossover(p1, p2):
    offspring = p1.copy()
    point = np.random.randint(0, len(p1))
    if np.random.rand() < 0.5:
        offspring[:point] = p2[:point]
    else:
        offspring[point:] = p2[point:]
    return offspring

def mutate(bitstring, mutation_rate=1):
    """ Mutate a bitstring with a given mutation rate. """
    for i in range(len(bitstring)):
        if np.random.rand() < mutation_rate / len(bitstring):
            bitstring[i] = 1 - bitstring[i]
    return bitstring

def count_duplicates(candidates):
    """ Count duplicates in the population. """
    for individual in candidates:
        yield sum(np.array_equal(individual, other) for other in candidates)

def ga(args):
    """ Run a (5+1) genetic algorithm with the given crossover function and mutation rate. """
    n, pop_size, crossover_function, mutation_rate, crossover_rate, fitness_func, max_gen = args
    population = [np.random.randint(0, 2, n) for _ in range(pop_size)]
    success_crossover = 0
    fit_hist = []
    info_hist = []
    results = [fitness_func(ind) for ind in population]
    fitnesses = [r[0] for r in results]
    infos = [r[1] for r in results]
    for generations in range(max_gen):
        #print(generations)
        parents_indices = np.random.choice(pop_size, 2, replace=False)
        parents = [population[parents_indices[0]], population[parents_indices[1]]]
        #print(np.sum(np.abs(parents[0] - parents[1])))
        if crossover_function and np.random.rand() < crossover_rate:
            offspring = crossover_function(parents[0], parents[1])
        else:
            offspring = np.copy(parents[0])  # for mutation only

        fit_hist.append(max(fitnesses))
        info_hist.append(infos[np.argmax(fitnesses)])
        min_fitness = min(fitnesses)
        min_fitness_individual_index = np.where(fitnesses == min_fitness)[0]

        #if(fitness_func(offspring) >= min_fitness):
        #    success_crossover += 1
        
        offspring = mutate(offspring, mutation_rate=mutation_rate)
        offspring_fitness, offspring_info = fitness_func(offspring)
        
        if offspring_fitness > min_fitness:
            #print(population + [offspring])
            inferior = np.random.randint(0, len(min_fitness_individual_index))
            population[min_fitness_individual_index[inferior]] = offspring
            fitnesses[min_fitness_individual_index[inferior]] = offspring_fitness
            infos[min_fitness_individual_index[inferior]] = offspring_info
        elif (offspring_fitness == min_fitness):
            #print(population + [offspring])
            pop_duplicates = np.array(list(count_duplicates([population[i] for i in min_fitness_individual_index] + [offspring])))
            argmax_dup = np.where(pop_duplicates == max(pop_duplicates))[0]
            #print(population + [offspring])
            inferior = random.choice(range(len(argmax_dup) - 1))
            population[min_fitness_individual_index[argmax_dup[inferior]]] = offspring
            infos[min_fitness_individual_index[argmax_dup[inferior]]] = offspring_info
        #print(generations,max(fitnesses))
    #print(success_crossover / generations)
    return fit_hist, info_hist

def ea(args):
    """ Run a (1+1) Evolutionary Algorithm. """
    n, crossover_function, mutation_rate, fitness_func, max_gen = args
    parent = np.random.randint(0, 2, n)
    max_fit, old_info = fitness_func(parent)
    fit_hist = []
    info_hist = []
    for generation in range(max_gen):
        offspring = mutate(np.copy(parent), mutation_rate=mutation_rate)
        fit, info= fitness_func(offspring)
        if fit >= max_fit:
            parent = offspring
            max_fit = fit
            old_info = info
        if max_fit == n:
            break
        fit_hist.append(max_fit)
        info_hist.append(old_info)
    #print(generations)
    return fit_hist ,info_hist

def calc_utilization(results):
    mean_util = []
    for i in range(max_gen):
        mean_util.append([r[1][i]['linkage_utilization'] for r in results])
    mean_util= np.mean(mean_util, axis=1).T
    return mean_util

if __name__ == '__main__':
    runs = 1000
    num_workers = 4
    # Mutation rates
    n = 4
    k = 4
    size = n * k
    max_gen = 200
    mutation_rates = np.linspace(1, 2, 9)
    crossover_rate = 1
    mr = 1
    prob_size = list(range(5, 60, 5))
    pop_size = [2, 5, 10, 20]

    L = Linkage(n, k, 1)
    L.seed(2)
    L.reset()
    mean_fit_specials = []
    mean_fit_uniforms = []
    std_specials = []
    std_uniforms = []
    mean_util_specials = []
    mean_util_uniforms = []



    test_func = L.step
    for pop in pop_size:
        workload = []
        for i in range(runs):
            workload.append((size, pop,special_crossover, mr, crossover_rate, test_func, max_gen))
        results = perform_mpi_function(workload, num_workers, ga)
        fit_special = np.array([r[0] for r in results])
        mean_fit_special = np.mean(fit_special, axis=0)
        std_special = np.std(fit_special, axis=0)
        mean_util_special = calc_utilization(results)
        mean_fit_specials.append(mean_fit_special)
        std_specials.append(std_special)
        mean_util_specials.append(mean_util_special)
        print(1)

        workload = []
        for i in range(runs):
            workload.append((size, pop, uniform_crossover, mr, crossover_rate, test_func, max_gen))
        results = perform_mpi_function(workload, num_workers, ga)
        fit_uniform = np.array([r[0] for r in results])
        mean_fit_uniform = np.mean(fit_uniform, axis=0)
        std_uniform = np.std(fit_uniform, axis=0)
        mean_util_uniform = calc_utilization(results)
        mean_fit_uniforms.append(mean_fit_uniform)
        std_uniforms.append(std_uniform)
        mean_util_uniforms.append(mean_util_uniform)
        print(2)

    workload = []
    for i in range(runs):
        workload.append((size, None, mr, test_func, max_gen))
    results = perform_mpi_function(workload, num_workers, ea)
    fit_mutation_only = np.array([r[0] for r in results])
    mean_fit_mutation_only = np.mean(fit_mutation_only, axis=0)
    std_mutation = np.std(fit_mutation_only, axis=0)
    mean_util_mutation_only = calc_utilization(results)

    

    #print(mean_util_special)
    #print(mean_util_uniform)
    #print(mean_util_mutation_only)
    # Run simulations and collect data
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(max_gen), mean_fit_specials[0], color='red', label='Block Uniform Crossover')
    plt.plot(range(max_gen), mean_fit_uniforms[0], color='blue', label='Uniform Crossover')
    plt.plot(range(max_gen), mean_fit_mutation_only, color='black', label='Mutation Only')
    plt.plot(range(max_gen), mean_fit_specials[1], color='red', linestyle='dashed', label='Block Uniform Crossover5')
    plt.plot(range(max_gen), mean_fit_uniforms[1], color='blue', linestyle='dashed', label='Uniform Crossover5')
    plt.plot(range(max_gen), mean_fit_specials[2], color='red', linestyle='dotted', label='Block Uniform Crossover10')
    plt.plot(range(max_gen), mean_fit_uniforms[2], color='blue', linestyle='dotted', label='Uniform Crossover10')
    plt.plot(range(max_gen), mean_fit_specials[3], color='red', linestyle='dashdot', label='Block Uniform Crossover20')
    plt.plot(range(max_gen), mean_fit_uniforms[3], color='blue', linestyle='dashdot', label='Uniform Crossover20')
    #plt.plot(prob_size, avg_generations_uniform_onemax, label='Uniform Crossover with ONEMAX')
    #plt.plot(prob_size, avg_generations_uniform_onemax_half, label='Uniform Crossover with ONEMAX and half problem size')
    #plt.plot(prob_size, avg_generations_mutation_only_onemax, label='Mutation only with ONEMAX')
    #plt.plot(prob_size, avg_generations_mutation_only_onemax_half, label='Mutation only with ONEMAX and half problem size')
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness')
    plt.title('')
    plt.legend()
    plt.savefig('linkage.png')

    for i in range(k):
        plt.figure(figsize=(12, 6))
        plt.plot(range(max_gen), mean_util_special[i], color='red', label='Block Uniform Crossover')
        plt.plot(range(max_gen), mean_util_uniform[i], color='blue', label='Uniform Crossover')
        plt.plot(range(max_gen), mean_util_mutation_only[i], color='black', label='Mutation Only')
        plt.xlabel('Generations')
        plt.ylabel('Average Fitness')
        plt.title('')
        plt.legend()
        plt.savefig('linkage_level' + str(i + 1) + '.png')