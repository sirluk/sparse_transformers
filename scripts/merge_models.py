from logging import LogRecord
import sys
sys.path.insert(0,'..')

import argparse
import ruamel.yaml as yaml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_diff_adv import AdvDiffModel
from src.models.model_task import TaskModel
from src.models.model_diff_modular import ModularDiffModel
from src.models.model_heads import ClfHead
from src.models.model_base import BaseModel
from src.model_functions import train_head, merge_adv_models, merge_modular_model, merge_models
from src.adv_attack import adv_attack
from src.data_handler import get_data
from src.model_functions import generate_embeddings
from src.utils import get_logger_custom, get_callables


def merge_adv_models_wrapper(cp_gender, cp_age, cp_base = None, mean_diff_weights = True, mean_ignore_zero = False):
    model_gender = AdvDiffModel.load_checkpoint(cp_gender)
    model_age = AdvDiffModel.load_checkpoint(cp_age)
    model = merge_adv_models(
        model_gender,
        model_age,
        mean_diff_weights = mean_diff_weights,
        mean_ignore_zero = mean_ignore_zero
    )
    if cp_base is not None:
        model_task = TaskModel.load_checkpoint(cp_base)
        model = merge_models(model_task.encoder, model.encoder, mean = False)
    return BaseModel(model.model_name, model.state_dict())


def merge_modular_models_wrapper(cp_modular, mean_diff_weights = True, mean_ignore_zero = False):
    modular_model = ModularDiffModel.load_checkpoint(cp_modular)
    model = merge_modular_model(modular_model, mean_diff_weights = mean_diff_weights, mean_ignore_zero = mean_ignore_zero)
    return BaseModel(modular_model.model_name, model.state_dict())



DEBUG = True
GPU_ID = 0
SEED = 0
DS = "pan16"
PCT = 0.1
MEAN = True
MEAN_IGNORE_ZERO = False

# CP = {
#     "task_model": f"/share/home/lukash/pan16/bertl4/cp/task-baseline-bert_uncased_L-4_H-256_A-4-64-2e-05-weighted_loss_prot-seed{SEED}.pt",
#     "adv_gender": f"/share/home/lukash/pan16/bertl4/cp_cp_init/adverserial-diff_pruning_{PCT}-bert_uncased_L-4_H-256_A-4-64-2e-05-cp_init-weighted_loss_prot-gender-seed{SEED}.pt",
#     "adv_age": f"/share/home/lukash/pan16/bertl4/cp_cp_init/adverserial-diff_pruning_{PCT}-bert_uncased_L-4_H-256_A-4-64-2e-05-cp_init-weighted_loss_prot-age-seed{SEED}.pt"
# }
CP = {
    "modular_model": f"/share/home/lukash/pan16/bertl4/cp_modular/modular-diff_pruning_{PCT}-freeze_task_head-bert_uncased_L-4_H-256_A-4-64-2e-05-weighted_loss_prot-gender_age-seed{SEED}.pt"
    # "modular_model": f"/share/home/lukash/pan16/bertl4/cp_modular/modular-diff_pruning_{PCT}-adv_task_head-bert_uncased_L-4_H-256_A-4-64-2e-05-weighted_loss_prot-gender_age-seed{SEED}.pt"
    # "modular_model": f"/share/home/lukash/pan16/bertl4/cp_modular/modular-diff_pruning_{PCT}-bert_uncased_L-4_H-256_A-4-64-2e-05-weighted_loss_prot-gender_age-seed{SEED}.pt"
}

LOG_DIR = f"logs_merged_masks_{DS}"

LOGGER_NAME = [
    "DEBUG" if DEBUG else None,
    "mod" if ("modular_model" in CP.keys()) else "adv",
    "adv_task_head" if ("modular_model" in CP.keys()) and ("adv_task_head" in CP["modular_model"]) else None,
    "frozen_task_head" if ("modular_model" in CP.keys()) and ("freeze" in CP["modular_model"]) else None,
    str(PCT),
    "avg" if MEAN else "additive",
    "ignore_zero" if MEAN and MEAN_IGNORE_ZERO else None,
    f"seed{SEED}"
]
LOGGER_NAME = "_".join([x for x in LOGGER_NAME if x is not None])


DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"


def main():

    torch.manual_seed(SEED)
    print(f"torch.manual_seed({SEED})")

    # model = merge_adv_models_wrapper(
    #     CP["adv_gender"],
    #     CP["adv_age"],
    #     mean_diff_weights=MEAN,
    #     mean_ignore_zero=MEAN_IGNORE_ZERO
    # ) # , cp_base=CP["task_model"]
    model = merge_modular_models_wrapper(
        CP["modular_model"],
        mean_diff_weights=MEAN,
        mean_ignore_zero=MEAN_IGNORE_ZERO
    ) # , cp_base=CP["task_model"]

    # # TEMP - for debugging
    # from src.model_functions import get_param_from_name
    
    # samples = []
    # for n, p in model.encoder.named_parameters():
    #     pg_diff = get_param_from_name(model_gender.encoder, n)
    #     pa_diff = get_param_from_name(model_age.encoder, n)
    #     if pg_diff.flatten()[1] != pa_diff.flatten()[1]:
    #         samples.append((n))

    # n = samples[-1]
    # p = get_param_from_name(model.encoder, n)
    # pg = get_param_from_name(model_gender.encoder, n)
    # pa = get_param_from_name(model_age.encoder, n)
    # import IPython; IPython.embed(); exit(1)
    # # TEMP - for debugging

    model.to(DEVICE)
    model.eval()

    with open("../cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{DS}"
    args_train = argparse.Namespace(**cfg["train_config"], **cfg[data_cfg], **cfg["model_config"])
    args_attack = argparse.Namespace(**cfg["adv_attack"])

    train_logger = get_logger_custom(
        log_dir=f"../{LOG_DIR}",
        logger_name=LOGGER_NAME
    )

    train_loader, val_loader, num_labels, num_labels_protected_list, protected_key_list, protected_class_weights_list = get_data(
        args_train = args_train,
        use_all_attr = True,
        compute_class_weights = args_train.weighted_loss_protected,
        device = DEVICE,
        debug = DEBUG
    )

    train_data = generate_embeddings(model, train_loader, forward_fn = lambda m, x: m._forward(**x))
    val_data = generate_embeddings(model, val_loader, forward_fn = lambda m, x: m._forward(**x))

    # Training Loop for task
    task_head = ClfHead(
        hid_sizes=[model.in_size_heads]*(args_train.task_n_hidden+1),
        num_labels=num_labels,
        dropout=args_train.task_dropout
    )
    task_head.to(DEVICE)

    del model

    ds_train = TensorDataset(train_data[0], train_data[1])
    ds_val = TensorDataset(val_data[0], val_data[1])
    train_loader = DataLoader(ds_train, shuffle=True, batch_size=args_attack.attack_batch_size, drop_last=False)
    val_loader = DataLoader(ds_val, shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False)

    loss_fn, pred_fn, metrics = get_callables(num_labels)

    task_head = train_head(
        head = task_head,
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        optim = AdamW,
        num_epochs = args_attack.num_epochs,
        lr = args_attack.learning_rate,
        cooldown = args_attack.cooldown,
        desc = "task_eval"
    )

    for i, (num_lbl_prot, prot_k, prot_w) in enumerate(zip(num_labels_protected_list, protected_key_list, protected_class_weights_list)):

        label_idx = i+2

        ds_train = TensorDataset(train_data[0], train_data[label_idx])
        ds_val = TensorDataset(val_data[0], val_data[label_idx])
        train_loader = DataLoader(ds_train, shuffle=True, batch_size=args_attack.attack_batch_size, drop_last=False)
        val_loader = DataLoader(ds_val, shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False)

        loss_fn, pred_fn, metrics = get_callables(num_lbl_prot, prot_w)

        adv_attack(
            train_loader = train_loader,
            val_loader = val_loader,
            logger = train_logger,
            loss_fn = loss_fn,
            pred_fn = pred_fn,
            metrics = metrics,
            num_labels = num_lbl_prot,
            adv_n_hidden = args_attack.adv_n_hidden,
            adv_count = args_attack.adv_count,
            adv_dropout = args_attack.adv_dropout,
            num_epochs = args_attack.num_epochs,
            lr = args_attack.learning_rate,
            cooldown = args_attack.cooldown,
            create_hidden_dataloader = False,
            device = DEVICE,
            logger_suffix = f"adv_attack_{prot_k}"
        )


if __name__ == "__main__":
    main()