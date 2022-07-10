import subprocess

# https://stackoverflow.com/questions/30686295/how-do-i-run-multiple-subprocesses-in-parallel-and-wait-for-them-to-finish-in-py

for i in range(5):
    subprocess.call(f'python3 main.py --modular=True --gpu_id=3 --seed={i}', shell=True)


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