import subprocess
import os

debug = True

suffix = " --debug" if debug else ""

folder = "/share/home/lukash/checkpoints_bert_L4/seed{}"
experiment_name = "bert_uncased_L-4_H-256_A-4-fixmask0.1-task.pt"

for i in range(5):
    filepath = os.path.join(folder.format(i), experiment_name)
    subprocess.call(f'python3 main.py --adv --gpu_id=7 --seed=0' + suffix, shell=True)


# from subprocess import Popen

# unique_commands = [
#     'python3 main.py --baseline=True --modular=True --gpu_id=1 --seed={} --debug=True',
#     'python3 main.py --baseline=True --gpu_id=1 --seed={} --debug=True'
# ]
# commands = []
# for c in unique_commands:
#     commands.extend([c.format(i) for i in range(5)])

# n = 2 #the number of parallel processes you want
# for j in range(max(int(len(commands)/n), 1)):
#     procs = [subprocess.Popen(i, shell=True) for i in commands[j*n: min((j+1)*n, len(commands))] ]
#     for p in procs:
#         p.wait()