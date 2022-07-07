import subprocess


subprocess.call('python3 main.py --baseline=True --modular=True --gpu_id=6 --seed=2', shell=True)

subprocess.call('python3 main.py --baseline=True --modular=True --gpu_id=6 --seed=3', shell=True)

subprocess.call('python3 main.py --baseline=True --modular=True --gpu_id=6 --seed=4', shell=True)