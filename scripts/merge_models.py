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
from src.model_functions import train_head, merge_adv_models, merge_modular_model
from src.adv_attack import adv_attack
from src.data_handler import get_data
from src.model_functions import generate_embeddings
from src.utils import get_logger_custom, get_callables, get_param_from_name

DEBUG = False
GPU_ID = 0
SEED = 0
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
DS = "pan16"
LOG_DIR = "logs_merged_masks"
LOGGER_NAME = "adv_seed{}".format(SEED)
MODEL_ADV_CLS = AdvDiffModel
MODEL_TASK_CLS = TaskModel
CP = {
    "task_model": f"/share/home/lukash/pan16/bertl4/cp/task-baseline-bert_uncased_L-4_H-256_A-4-64-2e-05-weighted_loss_prot-seed{SEED}.pt",
    "adv_gender": f"/share/home/lukash/pan16/bertl4/cp_cp_init/adverserial-diff_pruning_0.05-bert_uncased_L-4_H-256_A-4-64-2e-05-cp_init-weighted_loss_prot-gender-seed{SEED}.pt",
    "adv_age": f"/share/home/lukash/pan16/bertl4/cp_cp_init/adverserial-diff_pruning_0.05-bert_uncased_L-4_H-256_A-4-64-2e-05-cp_init-weighted_loss_prot-age-seed{SEED}.pt"
}
# CP = {
#     "modular_model": f"/share/home/lukash/pan16/bertl4/cp_special/modular-diff_pruning_0.1-bert_uncased_L-4_H-256_A-4-64-2e-05-weighted_loss_prot-gender_age-seed{SEED}-separate_optim.pt"
# }


def merge_adv_models_wrapper(cp_gender, cp_age, cp_base = None):

    if cp_base is None:
        model_gender = AdvDiffModel.load_checkpoint(cp_gender)
        model_age = AdvDiffModel.load_checkpoint(cp_age)
        model = merge_adv_models(model_gender, model_age)
        return BaseModel(model_gender.model_name, model.state_dict())
    else:
        model_gender = AdvDiffModel.load_checkpoint(cp_gender, remove_parametrizations=True)
        model_age = AdvDiffModel.load_checkpoint(cp_age, remove_parametrizations=True)
        model_task = TaskModel.load_checkpoint(cp_base)
        base_weights = model_task.encoder
        with torch.no_grad():
            for n, p in base_weights.named_parameters():
                p_gender = get_param_from_name(model_gender.encoder, n)
                p_age = get_param_from_name(model_age.encoder, n)
                p = p_gender + p_age - p
        return BaseModel(model_gender.model_name, base_weights.state_dict())


def merge_modular_models_wrapper(cp_modular):
    modular_model = ModularDiffModel.load_checkpoint(cp_modular)
    model = merge_modular_model(modular_model)
    return BaseModel(modular_model.model_name, model.state_dict())


def main():

    torch.manual_seed(SEED)
    print(f"torch.manual_seed({SEED})")

    model = merge_adv_models_wrapper(CP["adv_gender"], CP["adv_age"]) # , cp_base=CP["task_model"]
    # model = merge_modular_models_wrapper(CP["modular_model"]) # , cp_base=CP["task_model"]

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
    ds_train = TensorDataset(train_data[0], train_data[1])
    ds_val = TensorDataset(val_data[0], val_data[1])
    train_loader = DataLoader(ds_train, shuffle=True, batch_size=args_attack.attack_batch_size, drop_last=False)
    val_loader = DataLoader(ds_val, shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False)

    task_head = ClfHead(
        hid_sizes=[model.in_size_heads]*(args_train.task_n_hidden+1),
        num_labels=num_labels,
        dropout=args_train.task_dropout
    )
    task_head.to(DEVICE)

    loss_fn, pred_fn, metrics = get_callables(num_labels)

    task_head = train_head(
        trainer = model,
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
            trainer = model,
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
            batch_size = args_attack.attack_batch_size,
            label_idx = label_idx,
            logger_suffix = f"adv_attack_{prot_k}"
        )


if __name__ == "__main__":
    main()