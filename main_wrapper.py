import subprocess
import os

n = 5
debug = False

suffix = ' --debug' if debug else ''

# folder = "/share/home/lukash/checkpoints_bert_L4/seed{}"
# experiment_name = "bert_uncased_L-4_H-256_A-4-fixmask0.1-task.pt"

# for i in range(n):
#     # filepath = os.path.join(folder.format(i), experiment_name)
#     subprocess.call(f'python3 main.py --adv --gpu_id=7 --seed={i}' + suffix, shell=True)

# PAN16

subprocess.call('python3 main.py --modular --gpu_id=3 --seed=0 --modular_adv_task_head=False --fixmask_pct=0.01 --model_name="bert-base-uncased" --ds="pan16"' + suffix, shell=True)

subprocess.call('python3 main.py --modular --gpu_id=3 --seed=0 --modular_adv_task_head=False --fixmask_pct=0.02 --model_name="bert-base-uncased" --ds="pan16"' + suffix, shell=True)

subprocess.call('python3 main.py --baseline --modular --gpu_id=3 --seed=0 --modular_adv_task_head=False --model_name="bert-base-uncased" --ds="pan16"' + suffix, shell=True)

subprocess.call('python3 main.py --adv --gpu_id=3 --seed=0 --fixmask_pct=0.01 --model_name="bert-base-uncased" --ds="pan16"' + suffix, shell=True)

subprocess.call('python3 main.py --gpu_id=3 --seed=0 --fixmask_pct=0.01 --model_name="bert-base-uncased" --ds="pan16"' + suffix, shell=True)

subprocess.call('python3 main.py --adv --gpu_id=3 --seed=0 --fixmask_pct=0.02 --model_name="bert-base-uncased" --ds="pan16"' + suffix, shell=True)

subprocess.call('python3 main.py --gpu_id=3 --seed=0 --fixmask_pct=0.02 --model_name="bert-base-uncased" --ds="pan16"' + suffix, shell=True)

subprocess.call('python3 main.py --baseline --gpu_id=3 --seed=0 --model_name="bert-base-uncased" --ds="pan16"' + suffix, shell=True)

subprocess.call('python3 main.py --baseline --adv --gpu_id=3 --seed=0 --model_name="bert-base-uncased" --ds="pan16"' + suffix, shell=True)

# BIOS

subprocess.call('python3 main.py --modular --gpu_id=3 --seed=0 --modular_adv_task_head=False --fixmask_pct=0.01 --model_name="bert-base-uncased" --ds="bios"' + suffix, shell=True)

subprocess.call('python3 main.py --modular --gpu_id=3 --seed=0 --modular_adv_task_head=False --fixmask_pct=0.02 --model_name="bert-base-uncased" --ds="bios"' + suffix, shell=True)

subprocess.call('python3 main.py --baseline --modular --gpu_id=3 --seed=0 --modular_adv_task_head=False --model_name="bert-base-uncased" --ds="bios"' + suffix, shell=True)

subprocess.call('python3 main.py --adv --gpu_id=3 --seed=0 --fixmask_pct=0.01 --model_name="bert-base-uncased" --ds="bios"' + suffix, shell=True)

subprocess.call('python3 main.py --gpu_id=3 --seed=0 --fixmask_pct=0.01 --model_name="bert-base-uncased" --ds="bios"' + suffix, shell=True)

subprocess.call('python3 main.py --adv --gpu_id=3 --seed=0 --fixmask_pct=0.02 --model_name="bert-base-uncased" --ds="bios"' + suffix, shell=True)

subprocess.call('python3 main.py --gpu_id=3 --seed=0 --fixmask_pct=0.02 --model_name="bert-base-uncased" --ds="bios"' + suffix, shell=True)

subprocess.call('python3 main.py --baseline --gpu_id=3 --seed=0 --model_name="bert-base-uncased" --ds="bios"' + suffix, shell=True)

subprocess.call('python3 main.py --baseline --adv --gpu_id=3 --seed=0 --model_name="bert-base-uncased" --ds="bios"' + suffix, shell=True)
