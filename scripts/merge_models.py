import sys
sys.path.insert(0,'..')

import argparse
import ruamel.yaml as yaml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_diff_adv import AdvDiffModel
from src.models.model_task import TaskModel
from src.models.model_heads import ClfHead
from src.model_functions import merge_diff_models, train_head
from src.adv_attack import adv_attack
from src.data_handler import get_data
from src.model_functions import generate_embeddings
from src.utils import get_logger_custom, get_callables

DEBUG = False
GPU_ID = 0
SEED = 0
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
DS = "pan16"
LOG_DIR = "logs_custom"
LOGGER_NAME = "merged_mask_model_seed0"
MODEL_ADV_CLS = AdvDiffModel
MODEL_TASK_CLS = TaskModel
CP = {
    "task": "/share/home/lukash/pan16/bertl4/cp/task-baseline-bert_uncased_L-4_H-256_A-4-64-2e-05-seed{}.pt".format(SEED),
    "gender": "/share/home/lukash/pan16/bertl4/cp_init/task_baseline/cp/adverserial-diff_pruning_0.1-bert_uncased_L-4_H-256_A-4-64-2e-05-cp_init-weighted_loss_prot-gender-seed{}.pt".format(SEED),
    "age": "/share/home/lukash/pan16/bertl4/cp_init/task_baseline/cp/adverserial-diff_pruning_0.1-bert_uncased_L-4_H-256_A-4-64-2e-05-cp_init-weighted_loss_prot-age-seed{}.pt".format(SEED)
}


def main():

    torch.manual_seed(SEED)
    print(f"torch.manual_seed({SEED})")

    model_gender = MODEL_ADV_CLS.load_checkpoint(CP["gender"])
    model_age = MODEL_ADV_CLS.load_checkpoint(CP["age"])
    if "task" in CP:
        model_task = MODEL_TASK_CLS.load_checkpoint(CP["task"])
        model = merge_diff_models([model_gender, model_age], base_model=model_task)
    else:
        model = merge_diff_models([model_gender, model_age])

    # # TEMP - for debugging
    # from src.model_functions import get_param_from_name
    
    # samples = []
    # for n, p in model_task.encoder.named_parameters():
    #     _n = n.split(".")
    #     np = ".".join(_n[:-1] + ["parametrizations", _n[-1], "original"])
    #     np_diff = ".".join(_n[:-1] + ["parametrizations", _n[-1], "0", "diff_weight"])
    #     pg_diff = get_param_from_name(model_gender.encoder, np_diff)
    #     pa_diff = get_param_from_name(model_age.encoder, np_diff)
    #     if p.flatten()[1] != pg_diff.flatten()[1] != pa_diff.flatten()[1]:
    #         samples.append((n, np, np_diff))

    # n, np, np_diff = samples[-1]
    # p = get_param_from_name(model_task.encoder, n)
    # pg = get_param_from_name(model_gender.encoder, np)
    # pa = get_param_from_name(model_age.encoder, np)
    # pg_diff = get_param_from_name(model_gender.encoder, np_diff)
    # pa_diff = get_param_from_name(model_age.encoder, np_diff)
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
            label_idx = label_idx,
            logger_suffix = f"adv_attack_{prot_k}"
        )


if __name__ == "__main__":
    main()