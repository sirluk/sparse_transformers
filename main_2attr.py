import argparse
import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.model_diff_modular_2attr import ModularDiffModel_2attr
from src.adv_attack import adv_attack
from src.data_handler import get_data_loader
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    get_data,
    get_logger,
    get_callables,
    set_optional_args,
    get_num_labels
)

from typing import Tuple


def get_data(args_train: argparse.Namespace, debug: bool = False) -> Tuple[DataLoader, DataLoader, int, int]:

    num_labels = get_num_labels(args_train.labels_task_path)
    num_labels_protected = get_num_labels(args_train.labels_protected_path)
    num_labels_protected2 = get_num_labels(args_train.labels_protected_path2)
    tokenizer = AutoTokenizer.from_pretrained(args_train.model_name)
    train_loader = get_data_loader(
        task_key = args_train.task_key,
        protected_key = [args_train.protected_key, args_train.protected_key2],
        text_key = args_train.text_key,
        tokenizer = tokenizer,
        data_path = args_train.train_pkl,
        labels_task_path = args_train.labels_task_path,
        labels_prot_path = [args_train.labels_protected_path, args_train.labels_protected_path2],
        batch_size = args_train.batch_size,
        max_length = 200,
        debug = debug
    )
    val_loader = get_data_loader(
        task_key = args_train.task_key,
        protected_key = [args_train.protected_key, args_train.protected_key2],
        text_key = args_train.text_key,
        tokenizer = tokenizer,
        data_path = args_train.val_pkl,
        labels_task_path = args_train.labels_task_path,
        labels_prot_path = [args_train.labels_protected_path, args_train.labels_protected_path2],
        batch_size = args_train.batch_size,
        max_length = 200,
        shuffle = False,
        debug = debug
    )
    return train_loader, val_loader, num_labels, num_labels_protected, num_labels_protected2


def train(device, train_loader, val_loader, num_labels, num_labels_protected, num_labels_protected2, train_logger, args_train, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected)
    loss_fn_protected2, pred_fn_protected2, metrics_protected2 = get_callables(num_labels_protected2)

    trainer = ModularDiffModel_2attr(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected,
        num_labels_protected2 = num_labels_protected2,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
        adv_task_head = args_train.modular_adv_task_head,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        loss_fn_protected = loss_fn_protected,
        pred_fn_protected = pred_fn_protected,
        metrics_protected = metrics_protected,
        loss_fn_protected2 = loss_fn_protected2,
        pred_fn_protected2 = pred_fn_protected2,
        metrics_protected2 = metrics_protected2,
        num_epochs_warmup = args_train.num_epochs_warmup,
        num_epochs_finetune = args_train.num_epochs_finetune,
        num_epochs_fixmask = args_train.num_epochs_fixmask,
        alpha_init = args_train.alpha_init,
        concrete_samples = args_train.concrete_samples,
        concrete_lower = args_train.concrete_lower,
        concrete_upper = args_train.concrete_upper,
        structured_diff_pruning = args_train.structured_diff_pruning,
        adv_lambda = args_train.adv_lambda,
        sparsity_pen = args_train.sparsity_pen,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_task_head = args_train.learning_rate_task_head,
        learning_rate_adv_head = args_train.learning_rate_adv_head,
        learning_rate_alpha = args_train.learning_rate_alpha,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        weight_decay = args_train.weight_decay,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        sparse_task = False,
        merged_cutoff = args_train.modular_merged_cutoff,
        merged_min_pct = args_train.modular_merged_min_pct,
        fixmask_pct = args_train.fixmask_pct,
        seed = seed
    )
    trainer = ModularDiffModel_2attr.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--ds", type=str, default="bios", help="dataset")
    parser.add_argument("--debug", action="store_true", help="Whether to run on small subset for testing")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    parser.add_argument("--no_adv_attack", action="store_true", help="Set if you do not want to run adverserial attack after training")
    base_args, optional = parser.parse_known_args()

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{base_args.ds}"
    args_train = argparse.Namespace(**cfg["train_config"], **cfg[data_cfg], **cfg["model_config"])
    args_attack = argparse.Namespace(**cfg["adv_attack"])

    set_optional_args(args_train, optional)

    if base_args.debug:
        set_num_epochs_debug(args_train)
        set_num_epochs_debug(args_attack)
        set_dir_debug(args_train)

    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    device = get_device(not base_args.cpu, base_args.gpu_id)
    print(f"Device: {device}")

    train_loader, val_loader, num_labels, num_labels_protected, num_labels_protected2 = \
        get_data(args_train, debug=base_args.debug)
    
    train_logger = get_logger(
        False, False, True, args_train, base_args.debug, False, base_args.seed, f"2attr-{args_train.protected_key}"
    )

    print(f"Running {train_logger.logger_name}")

    trainer = train(device, train_loader, val_loader, num_labels, num_labels_protected, num_labels_protected2, train_logger, args_train, base_args.seed)

    if not base_args.no_adv_attack:
        loss_fn, pred_fn, metrics = get_callables(num_labels_protected)
        adv_attack(
            trainer = trainer,
            train_loader = train_loader,
            val_loader = val_loader,
            logger = train_logger,
            loss_fn = loss_fn,
            pred_fn = pred_fn,
            metrics = metrics,
            num_labels = num_labels_protected,
            adv_n_hidden = args_attack.adv_n_hidden,
            adv_count = args_attack.adv_count,
            adv_dropout = args_attack.adv_dropout,
            num_epochs = args_attack.num_epochs,
            lr = args_attack.learning_rate,
            batch_size = args_attack.attack_batch_size,
            cooldown = args_attack.cooldown,
            logger_suffix = f"adv_attack_unbiased"
        )
        trainer.set_debiased(False)
        adv_attack(
            trainer = trainer,
            train_loader = train_loader,
            val_loader = val_loader,
            logger = train_logger,
            loss_fn = loss_fn,
            pred_fn = pred_fn,
            metrics = metrics,
            num_labels = num_labels_protected,
            adv_n_hidden = args_attack.adv_n_hidden,
            adv_count = args_attack.adv_count,
            adv_dropout = args_attack.adv_dropout,
            num_epochs = args_attack.num_epochs,
            lr = args_attack.learning_rate,
            batch_size = args_attack.attack_batch_size,
            cooldown = args_attack.cooldown,
            logger_suffix = "adv_attack_biased"
        )


if __name__ == "__main__":

    main()

