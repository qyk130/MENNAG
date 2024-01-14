import os
import sys
sys.path.append(os.getcwd() + '/src')
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import numpy as np
import gym
import multiprocessing as mp
import json
import ea
from nodes import Root
import time
import psutil
import random
import gc
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor, MPIPoolExecutor
from eval.eval import eval, EvalSummary
from configs import Configs
from tasks.task import get_task

DEBUG_FLAG = False
np.seterr(all='raise')

def get_ea(name):
    print(name)
    if (name == "speciation"):
        ea_instance = ea.EASpeciation(config)
    elif (name == "mapelite"):
        ea_instance = ea.EAMapelite(config)
    elif (name == "cma"):
        ea_instance = ea.EACMA(config)
    elif (name == "optimal_mixing"):
        import omea
        ea_instance = omea.OMEA(config)
    else:
        ea_instance = ea.EA(config)
    return ea_instance

def mpi_main(args, config, task):
    DEBUG_FLAG = args.debug
    ea = get_ea(config.ea)
    print(ea)
    reseedPeriod = int(args.reseed)
    taskNum = int(args.task_num)
    #np.random.seed(1)
    seed = np.random.randint(0, 2**32 - 1, size=(taskNum), dtype=np.uint32)
    seed = seed.tolist()
    while not ea.stop():
        if ((reseedPeriod > 0) and (i % reseedPeriod == 0)):
            for j in range(taskNum):
                seed[j] = random.randint(0, 2**32 - 1)
        print(seed)
        ea_time = time.time()
        pop = ea.ask()
        ea_time = time.time() - ea_time
        fitnesses = []
        workloads = []
        num_workers = int(args.num_workers) - 1
        prep_time = time.time()
        for j in range(len(pop)):
            workloads.append((
                    pop[j],
                    task,
                    seed,
                    args.debug,
                    config
                ))
        prep_time = time.time() - prep_time
        eval_time = time.time()
        results = perform_mpi_function(args.mpi, workloads, num_workers, eval)
        reducedResult = EvalSummary()
        reducedResult.reduce(results, 'pfit')
        ea.tell(reducedResult, args.task, seed)
        if (config.ea != 'cma'):
            ea.reproduce()
        eval_time = time.time() - eval_time
        if (ea.gen % 10 == 0):
            ea.results.store['evals'] *= 10
            ea.write_history(args.out_path)
        #print(ea.fitnesses)
        print('iter: {0} fit: {1}, Q: {2}, ea_time: {3}, prep_time: {4}, eval_time: {5}, max_depth:{6}'.format(
                ea.gen,
                ea.performance[0],
                #np.mean(reducedResult.get_res('Q')[0]),
                0,
                ea_time,
                prep_time,
                eval_time,
                ea.pop[0].maxDepth,
                #np.mean(reducedResult.get_res('pfit')[0])
            ))

def om_main(args, config, task):
    ea = get_ea(config.ea)
    reseedPeriod = int(args.reseed)
    taskNum = int(args.task_num)
    #np.random.seed(1)
    seed = np.random.randint(0, 2**32 - 1, size=(taskNum), dtype=np.uint32)
    seed = seed.tolist()
    evals = ea.ask(task, seed)
    while evals < config.max_evals:
        runtime = time.time()
        ea.ask(task, seed)
        results, current_evals = ea.reproduce()
        evals += current_evals
        runtime = time.time() - runtime
        print('eval: {0} fit: {1} pfit:{2} time: {3}'.format(
                evals,
                results.store['fit'][0],
                results.get_metric()[0],
                runtime
            ))
        ea.write_history(args.out_path)
    ea.end()
    ea.write_history(args.out_path)

def perform_mpi_function(mpi, workloads, num_workers, func):
    result_list = []
    if mpi:
        success = False
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
    else:
        for work in workloads:
            result_list.append(func(work))
    return result_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='Task name')
    parser.add_argument('-c', '--config', help='Configuration file')
    parser.add_argument('-g', '--generation', help='Generation number')
    parser.add_argument('-n', '--num_workers', help='Number of cores', default=1)
    parser.add_argument('-d', '--dir', help='Directory')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--reseed', default=-1)
    parser.add_argument('--task_num', default=1)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    args.out_path = args.dir + '/' + args.output
    with open(args.config) as configFile:
        configDict = json.load(configFile)
    config = Configs(configDict)
    print(args.mpi)
    config.mpi = args.mpi
    config.num_workers = int(args.num_workers) - 1
    config.generation = int(args.generation)
    task = get_task(args.task, config)
    if (config.ea == "optimal_mixing"):
        om_main(args, config, task)
    else:
        mpi_main(args, config, task)
