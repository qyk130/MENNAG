import subprocess
import os

#task = 'BipedalWalker-v3'
#setting = 'bipedal_direct'
tasks = [
        'Parity',
        #'ParitySmooth',
        #'MNIST_ACC',
        #'BipedalWalker-v3'
]
settings = [
            #'bipedal_direct_rom_leifos_ns',
            #'bipedal_direct_rom_leifos',
            #'bipedal_direct_rom_uniformcross',
            #'bipedal_direct_rom_uniformcross_ns',
            #'bipedal_direct_rom_nocross',
            #'parity_direct_gpom_ltfos',
            'parity10_direct_rom_leifos_ns',
            'parity10_direct_rom_leifos',
            #'parity_direct_cma',
            #'parity_direct_rom_uniformcross',
            #'parity_direct_rom_uniformcross_ns',
            #'parity_direct_rom_nocross'
            #'mnist_direct_rom_leifos_ns',
            #'mnist_direct_rom_leifos',
            #'mnist_direct_cma',
            #'mnist_direct_rom_uniformcross',
            #'mnist_direct_rom_uniformcross_ns',
            #'mnist_direct_rom_nocross'
]
gen = 2001
reseed = -1
taskNum = 1
for task in tasks:
    for setting in settings:
        if not (os.path.exists('out/'+task+setting)):
            print('makedir '+'out/'+task+setting)
            os.makedirs('out/'+task+setting)
        for i in range(0, 20):
            qsub_command = """qsub -v INDEX={0},TASK=\'{1}\',SETTING=\'{2}\',TASKNUM=\'{3}\',GEN=\'{4}\',RESEED=\'{5}\' -N {1}_{2}_{0} shell_mpi_batch.sh""".format(i, task, setting, taskNum, gen, reseed)
            print(qsub_command)
            e = subprocess.call(qsub_command, shell=True)
            if e is 1:
                print("failed")
