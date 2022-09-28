import subprocess

GPU_ID = 0
N = 5
MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4" # "bert-base-uncased" # "google/bert_uncased_L-4_H-256_A-4" # "google/bert_uncased_L-2_H-128_A-2"
FIXMASK_PCT_HIGH = 0.1
FIXMASK_PCT_LOW = 0.05
DS = "pan16"
PROT_KEY_IDX = None
CMD_IDX = [0,1,2,3,4,5,6,7,8,9,10]
DEBUG = False

prot_key_idx = "" if PROT_KEY_IDX is None else f" --prot_key_idx={PROT_KEY_IDX}"
debug = " --debug" if DEBUG else ""
for i in range(N):
    cmds = {
        0: f"python3 main.py --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --fixmask_pct={FIXMASK_PCT_LOW} --model_name={MODEL_NAME} --ds={DS}" + prot_key_idx + debug,
        1: f"python3 main.py --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --fixmask_pct={FIXMASK_PCT_HIGH} --model_name={MODEL_NAME} --ds={DS}" + prot_key_idx + debug,
        2: f"python3 main.py --baseline --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --model_name={MODEL_NAME} --ds={DS}" + prot_key_idx + debug,
        3: f"python3 main.py --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --fixmask_pct={FIXMASK_PCT_LOW} --model_name={MODEL_NAME} --ds={DS} --modular_sparse_task='False'" + prot_key_idx + debug,
        4: f"python3 main.py --modular --gpu_id={GPU_ID} --seed={i} --modular_adv_task_head=False --fixmask_pct={FIXMASK_PCT_HIGH} --model_name={MODEL_NAME} --ds={DS} --modular_sparse_task='False'" + prot_key_idx + debug,
        5: f"python3 main.py --adv --gpu_id={GPU_ID} --seed={i} --fixmask_pct={FIXMASK_PCT_LOW} --model_name={MODEL_NAME} --ds={DS}" + prot_key_idx + debug,
        6: f"python3 main.py --adv --gpu_id={GPU_ID} --seed={i} --fixmask_pct={FIXMASK_PCT_HIGH} --model_name={MODEL_NAME} --ds={DS}" + prot_key_idx + debug,
        7: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --ds={DS}" + prot_key_idx + debug,
        8: f"python3 main.py --gpu_id={GPU_ID} --seed={i} --fixmask_pct={FIXMASK_PCT_LOW} --model_name={MODEL_NAME} --ds={DS}" + debug,
        9: f"python3 main.py --gpu_id={GPU_ID} --seed={i} --fixmask_pct={FIXMASK_PCT_HIGH} --model_name={MODEL_NAME} --ds={DS}" + debug,
        10: f"python3 main.py --baseline --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --ds={DS}" + debug
    }
    for j in CMD_IDX:
        subprocess.call(cmds[j], shell=True)