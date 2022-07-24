import subprocess
import os

n = 5
debug = False

suffix = " --debug" if debug else ""

# folder = "/share/home/lukash/checkpoints_bert_L4/seed{}"
# experiment_name = "bert_uncased_L-4_H-256_A-4-fixmask0.1-task.pt"

for i in range(n):
    # filepath = os.path.join(folder.format(i), experiment_name)
    subprocess.call(f'python3 main.py --adv --gpu_id=7 --seed={i}' + suffix, shell=True)