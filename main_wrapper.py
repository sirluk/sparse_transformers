import subprocess
import os

GPU_ID = 0
N = 5
MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4" # "bert-base-uncased" # "google/bert_uncased_L-4_H-256_A-4" # "google/bert_uncased_L-2_H-128_A-2"
FIXMASK_PCT_HIGH = 0.1
FIXMASK_PCT_LOW = 0.05
DS = "bios"
PROT_KEY_IDX = 0
CMD_IDX = [0,1,2]
DEBUG = False
SUFFIX = " --debug" if DEBUG else ""


for i in range(N):
    cmds = {
        0: f"python3 main.py --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --fixmask_pct={FIXMASK_PCT_LOW} --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        1: f"python3 main.py --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --fixmask_pct={FIXMASK_PCT_HIGH} --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        2: f"python3 main.py --baseline --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        3: f"python3 main.py --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --fixmask_pct={FIXMASK_PCT_LOW} --model_name={MODEL_NAME} --ds={DS} --modular_sparse_task='False' --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        4: f"python3 main.py --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --fixmask_pct={FIXMASK_PCT_HIGH} --model_name={MODEL_NAME} --ds={DS} --modular_sparse_task='False' --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        5: f"python3 main.py --baseline --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --model_name={MODEL_NAME} --ds={DS} --modular_sparse_task='False' --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        6: f"python3 main.py --adv --gpu_id={GPU_ID} --seed={i} --fixmask_pct={FIXMASK_PCT_LOW} --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        7: f"python3 main.py --adv --gpu_id={GPU_ID} --seed={i} --fixmask_pct={FIXMASK_PCT_HIGH} --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        8: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        9: f"python3 main.py --gpu_id={GPU_ID} --seed={i} --fixmask_pct={FIXMASK_PCT_LOW} --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        10: f"python3 main.py --gpu_id={GPU_ID} --seed={i} --fixmask_pct={FIXMASK_PCT_HIGH} --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={PROT_KEY_IDX}" + SUFFIX,
        11: f"python3 main.py --baseline --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={PROT_KEY_IDX}" + SUFFIX
    }
    for j in CMD_IDX:
        subprocess.call(cmds[j], shell=True)