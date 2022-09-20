import subprocess
import os

n = 5
debug = False

suffix = " --debug" if debug else ""

# folder = '/share/home/lukash/checkpoints_bert_L4/seed{}'
# experiment_name = 'bert_uncased_L-4_H-256_A-4-fixmask0.1-task.pt'

for i in range(n):
    # subprocess.call(f"python3 main.py --modular --gpu_id=3 --seed={i} --modular_adv_task_head=False --fixmask_pct=0.05 --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16' --prot_key_idx=1" + suffix, shell=True)

    # subprocess.call(f"python3 main.py --modular --gpu_id=3 --seed={i} --modular_adv_task_head=False --fixmask_pct=0.1 --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16' --prot_key_idx=1" + suffix, shell=True)

    # subprocess.call(f"python3 main.py --baseline --modular --gpu_id=3 --seed={i} --modular_adv_task_head=False --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16' --prot_key_idx=1" + suffix, shell=True)

    # subprocess.call(f"python3 main.py --modular --gpu_id=1 --seed={i} --modular_adv_task_head=False --fixmask_pct=0.05 --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16' --modular_sparse_task='False' --prot_key_idx=1" + suffix, shell=True)

    # subprocess.call(f"python3 main.py --modular --gpu_id=1 --seed={i} --modular_adv_task_head=False --fixmask_pct=0.1 --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16' --modular_sparse_task='False' --prot_key_idx=1" + suffix, shell=True)

    # subprocess.call(f"python3 main.py --baseline --modular --gpu_id=1 --seed={i} --modular_adv_task_head=False --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16' --modular_sparse_task='False' --prot_key_idx=1" + suffix, shell=True)

    subprocess.call(f"python3 main.py --adv --gpu_id=0 --seed={i} --fixmask_pct=0.05 --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16' --prot_key_idx=1" + suffix, shell=True)

    subprocess.call(f"python3 main.py --adv --gpu_id=0 --seed={i} --fixmask_pct=0.1 --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16' --prot_key_idx=1" + suffix, shell=True)

    subprocess.call(f"python3 main.py --baseline --adv --gpu_id=0 --seed={i} --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16' --prot_key_idx=1" + suffix, shell=True)

    # subprocess.call(f"python3 main.py --gpu_id=2 --seed={i} --fixmask_pct=0.05 --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16'" + suffix, shell=True)

    # subprocess.call(f"python3 main.py --gpu_id=2 --seed={i} --fixmask_pct=0.1 --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16'" + suffix, shell=True)

    # subprocess.call(f"python3 main.py --baseline --gpu_id=2 --seed={i} --model_name='google/bert_uncased_L-4_H-256_A-4' --ds='pan16'" + suffix, shell=True)
