# import IPython; IPython.embed(); exit(1)

import ruamel.yaml as yaml
import argparse
from pathlib import Path
import torch

from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_diff_modular import ModularDiffModel
from src.models.model_diff_modular_legacy import ModularDiffModel as ModularDiffModelLegacy
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.training_logger import TrainLogger
from src.adv_attack import adv_attack
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    get_data,
    get_callables
)

torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="Whether to run on small subset for testing")
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--ds", type=str, default="bios", help="dataset")
    parser.add_argument("--cpu", type=bool, default=False, help="Run on cpu")
    base_args = parser.parse_args()

    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    device = get_device(not base_args.cpu, base_args.gpu_id)

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{base_args.ds}"
    args_train = argparse.Namespace(**cfg["adv_attack"], **cfg[data_cfg], **cfg["model_config"])

    if base_args.debug:
        args_train = set_num_epochs_debug(args_train)
        args_train = set_dir_debug(args_train)

    train_loader, eval_loader, num_labels, num_labels_protected = get_data(args_train, ds=base_args.ds, debug=base_args.debug)

    ### DEFINE MANUALLY
    cp_dir = "/share/home/lukash/checkpoints_bert_L4/seed0"
    # cp_dir = "checkpoints_bios_seed1"
    # cp = "bert_uncased_L-4_H-256_A-4-fixmask0.1-modular-sparse_task-merged_head.pt"
    cp = "bert_uncased_L-4_H-256_A-4-fixmask0.1-modular.pt"
    # cp = "bert_uncased_L-4_H-256_A-4-task_baseline.pt"
    # cp = "bert_uncased_L-4_H-256_A-4-fixmask0.05-task.pt"
    # cp = "bert_uncased_L-4_H-256_A-4-fixmask0.1-adv.pt"
    # cp = "bert_uncased_L-4_H-256_A-4-fixmask0.1-modular.pt"
    model_cls = ModularDiffModel
    ### DEFINE MANUALLY

    trainer = model_cls.load_checkpoint(f"{cp_dir}/{cp}")
    trainer.to(device)

    logger_name = "_".join([
        f"only_adv_attack_{cp}",
        str(args_train.batch_size),
        str(args_train.learning_rate),
        f"seed{base_args.seed}"
    ])
    train_logger = TrainLogger(
        log_dir = Path(args_train.log_dir),
        logger_name = logger_name,
        logging_step = args_train.logging_step
    )

    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected)

    print(f"running model {cp}")

    adv_attack(
        trainer = trainer,
        train_loader = train_loader,
        val_loader = eval_loader,
        logger = train_logger,
        loss_fn = loss_fn_protected,
        pred_fn = pred_fn_protected,
        metrics = metrics_protected,
        num_labels = num_labels_protected,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
        adv_dropout = args_train.adv_dropout,
        num_epochs = args_train.num_epochs,
        lr = args_train.learning_rate,
        cooldown = args_train.cooldown
    )

if __name__ == "__main__":

    main()


