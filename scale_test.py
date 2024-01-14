import subprocess

task = 'BipedalWalker-v3'
setting = 'bipedal_small'
#task = 'Retina'
#setting = 'retina_default'
gen = 1000
taskNum = 1
for i in range(0, 20):
    qsub_command = """qsub -v INDEX={0},TASK=\'{1}\',SETTING=\'{2}\',TASKNUM=\'{3}\',GEN=\'{4}\' -N {1}_{0} shell_mpi_batch.sh""".format(i, task, setting, taskNum, gen)
    print(qsub_command)
    e = subprocess.call(qsub_command, shell=True)
    if e is 1:
        print("failed")
