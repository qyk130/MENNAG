import matplotlib.pyplot as plt
import numpy as np
import os
from mpi4py.futures import MPICommExecutor, MPIPoolExecutor

#mpirun --oversubscribe -n 4 python -m mpi4py.futures addxor_test.py

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def perform_mpi_function(workloads, num_workers, func):
    success = False
    result_list = []
    if False:
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
    return np.sum([bitstring[i] * bitstring[i+1] * bitstring[i+2] * bitstring[i+3] * bitstring[i+4] * 5 for i in range(0, len(bitstring), 5)])

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
    block_size = 5
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
    pop_size = 20
    n, crossover_function, mutation_rate, crossover_rate, fitness_func = args
    population = [np.random.randint(0, 2, n) for _ in range(pop_size)]
    generations = 0
    success_crossover = 0
    fitnesses = [fitness_func(individual) for individual in population]
    while True:
        generations += 1
        parents_indices = np.random.choice(pop_size, 2, replace=False)
        parents = [population[parents_indices[0]], population[parents_indices[1]]]
        #print(np.sum(np.abs(parents[0] - parents[1])))
        if crossover_function and np.random.rand() < crossover_rate:
            offspring = crossover_function(parents[0], parents[1])
        else:
            print('error')
            offspring = np.copy(parents[0])  # for mutation only

        min_fitness = min(fitnesses)
        min_fitness_individual_index = np.where(fitnesses == min_fitness)[0]

        #if(fitness_func(offspring) >= min_fitness):
        #    success_crossover += 1
        
        offspring = mutate(offspring, mutation_rate=mutation_rate)
        offspring_fitness = fitness_func(offspring)
        
        if offspring_fitness > min_fitness:
            #print(population + [offspring])
            inferior = np.random.randint(0, len(min_fitness_individual_index))
            population[min_fitness_individual_index[inferior]] = offspring
            fitnesses[min_fitness_individual_index[inferior]] = offspring_fitness
        elif (offspring_fitness == min_fitness):
            #print(population + [offspring])
            pop_duplicates = np.array(list(count_duplicates([population[i] for i in min_fitness_individual_index] + [offspring])))
            argmax_dup = np.where(pop_duplicates == max(pop_duplicates))[0]
            #print(population + [offspring])
            inferior = np.random.randint(0, len(argmax_dup) - 1)
            population[min_fitness_individual_index[argmax_dup[inferior]]] = offspring

        if (max(fitnesses) == n or generations == 10000):
            break
        #print(generations,max(fitnesses))
    #print(success_crossover / generations)
    return generations

def ea(args):
    """ Run a (1+1) Evolutionary Algorithm. """
    n, crossover_function, mutation_rate, fitness_func = args
    parent = np.random.randint(0, 2, n)
    max_fit = fitness_func(parent)
    generations = 0
    while True:
        generations += 1
        offspring = mutate(np.copy(parent), mutation_rate=mutation_rate)
        fit = fitness_func(offspring)
        if fit >= max_fit:
            parent = offspring
            max_fit = fit
        if max_fit == n:
            break
    #print(generations)
    return generations

if __name__ == '__main__':
    runs = 100
    num_workers = 4
    mutation_rates = np.linspace(1, 3, 29)
    crossover_rate = 1
    mr = 1
    n = 100
    prob_size = list(range(5, 60, 5))
    avg_generations_special = []
    std_special = []
    avg_generations_uniform = []
    std_uniform = []
    avg_generations_mutation_only = []
    std_mutation = []
    avg_generations_uniform_onemax = []
    avg_generations_onepoint_onemax = []
    avg_generations_mutation_only_onemax = []
    for mr in mutation_rates:
        print(mr)
        workload = []
        for i in range(runs):
            workload.append((n, special_crossover, mr, crossover_rate, royal_road))
        results = perform_mpi_function(workload, num_workers, ga)
        std_special.append(np.std(results))
        avg_generations_special.append(sum(results) / runs)
        print(1)
        workload = []
        for i in range(runs):
            workload.append((n, uniform_crossover, mr, crossover_rate, royal_road))
        results = perform_mpi_function(workload, num_workers, ga)
        std_uniform.append(np.std(results))
        avg_generations_uniform.append(sum(results) / runs)
        print(2)
        workload = []
        for i in range(runs):
            workload.append((n, None, mr, royal_road))
        results = perform_mpi_function(workload, num_workers, ea)
        std_mutation.append(np.std(results))
        avg_generations_mutation_only.append(sum(results) / runs)
        print(3)
        '''
        workload = []
        for i in range(runs):
            workload.append((n, uniform_crossover, mr, crossover_rate, onemax))
        results = perform_mpi_function(workload, num_workers, ga)
        avg_generations_uniform_onemax.append(sum(results) / runs)
        print(4)

        workload = []
        for i in range(runs):
            workload.append((n, None,  mr, onemax))
        results = perform_mpi_function(workload, num_workers, ea)
        avg_generations_mutation_only_onemax.append(sum(results) / runs)
        print(5)
        '''

    # Run simulations and collect data
    # Plotting
    x = mutation_rates
    plt.figure(figsize=(12, 6))
    plt.errorbar(x, avg_generations_special, yerr=std_special, fmt='-o', label='Block Uniform Crossover')
    plt.errorbar(x, avg_generations_uniform, yerr=std_uniform, fmt='-v', label='Uniform Crossover')
    plt.errorbar(x, avg_generations_mutation_only, yerr=std_mutation, fmt='-s', label='Mutation Only')
    #plt.plot(prob_size, avg_generations_uniform_onemax, label='Uniform Crossover with ONEMAX')
    #plt.plot(prob_size, avg_generations_uniform_onemax_half, label='Uniform Crossover with ONEMAX and half problem size')
    #plt.plot(prob_size, avg_generations_mutation_only_onemax, label='Mutation only with ONEMAX')
    #plt.plot(prob_size, avg_generations_mutation_only_onemax_half, label='Mutation only with ONEMAX and half problem size')
    plt.xlabel('Problem size')
    plt.ylabel('Average Generations')
    plt.title('Average Generations vs Problem Size')
    plt.legend()
    plt.savefig('royal.png')