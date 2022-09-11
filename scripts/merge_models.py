import sys
sys.path.insert(0,'..')

import argparse
import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_diff_adv import AdvDiffModel
from src.models.model_task import TaskModel
from src.models.model_functions import merge_diff_models
from src.adv_attack import adv_attack
from src.utils import (
    get_data,
    get_logger_custom,
    get_callables,
    generate_embeddings
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DS = "pan16"
CP = {
    "task": "/share/home/lukash/pan16/bertl4/cp/bert_uncased_L-4_H-256_A-4-task_baseline-seed0.pt",
    "gender": "../checkpoints_pan16/bert_uncased_L-4_H-256_A-4-adv_fixmask0.1-cp_init-gender-seed0.pt",
    "age": "../checkpoints_pan16/bert_uncased_L-4_H-256_A-4-adv_fixmask0.1-cp_init-age-seed0.pt"
}

model_gender = AdvDiffModel.load_checkpoint(CP["gender"])
model_age = AdvDiffModel.load_checkpoint(CP["age"])
model_task = TaskModel.load_checkpoint(CP["task"])
model = merge_diff_models([model_gender, model_age], base_model=model_task)
model.to(DEVICE)
model.eval()

with open("../cfg.yml", "r") as f:
    cfg = yaml.safe_load(f)
data_cfg = f"data_config_{DS}"
args_train = argparse.Namespace(**cfg["train_config"], **cfg[data_cfg], **cfg["model_config"])
args_attack = argparse.Namespace(**cfg["adv_attack"])

train_logger = get_logger_custom(
    log_dir="../logs_custom",
    logger_name="merged_masks_model"
)

train_loader, val_loader, num_labels, num_labels_protected, num_labels_protected2 = \
    get_data(args_train, use_all_attr=True, debug=False)

train_data = generate_embeddings(model, train_loader, forward_fn = lambda m, x: m._forward(**x))
val_data = generate_embeddings(model, val_loader, forward_fn = lambda m, x: m._forward(**x))

# TODO: Training Loop for task
# ds_train = TensorDataset(train_data[0], train_data[1])
# ds_val = TensorDataset(val_data[0], val_data[1])
# train_loader = DataLoader(ds_train, shuffle=True, batch_size=args_attack.attack_batch_size, drop_last=False)
# val_loader = DataLoader(ds_val, shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False)

for i, (attr, num_labels) in enumerate(zip(args_train.protected_key, [num_labels_protected, num_labels_protected2])):

    ds_train = TensorDataset(train_data[0], train_data[i+2])
    ds_val = TensorDataset(val_data[0], val_data[i+2])
    train_loader = DataLoader(ds_train, shuffle=True, batch_size=args_attack.attack_batch_size, drop_last=False)
    val_loader = DataLoader(ds_val, shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False)

    loss_fn, pred_fn, metrics = get_callables(num_labels)

    adv_attack(
        trainer = model,
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        num_labels = num_labels,
        adv_n_hidden = args_attack.adv_n_hidden,
        adv_count = args_attack.adv_count,
        adv_dropout = args_attack.adv_dropout,
        num_epochs = args_attack.num_epochs,
        lr = args_attack.learning_rate,
        create_hidden_dataloader = False,
        logger_suffix = f"adv_attack_{attr}"
    )