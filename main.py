import argparse
import ruamel.yaml as yaml
import torch

from src.models.model_diff_modular import ModularDiffModel
from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.adv_attack import adv_attack
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    get_data,
    get_logger,
    get_callables
)


def train_diff_pruning_task(device, train_loader, val_loader, num_labels, train_logger, args_train):

    loss_fn, pred_fn, metrics = get_callables(num_labels)

    trainer = TaskDiffModel(
        model_name = args_train.model_name,
        num_labels = num_labels,
        dropout = args_train.task_dropout,
        n_hidden = args_train.task_n_hidden,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        num_epochs_finetune = args_train.num_epochs_finetune,
        num_epochs_fixmask = args_train.num_epochs_fixmask,
        alpha_init = args_train.alpha_init,
        concrete_samples = args_train.concrete_samples,
        concrete_lower = args_train.concrete_lower,
        concrete_upper = args_train.concrete_upper,
        structured_diff_pruning = args_train.structured_diff_pruning,
        sparsity_pen = args_train.sparsity_pen,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_head = args_train.learning_rate_task_head,
        learning_rate_alpha = args_train.learning_rate_alpha,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        weight_decay = args_train.weight_decay,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        cooldown = args_train.cooldown,
        fixmask_pct = args_train.fixmask_pct
    )
    trainer = TaskDiffModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_diff_pruning_adv(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected)

    trainer = AdvDiffModel(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
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
        fixmask_pct = args_train.fixmask_pct
    )
    trainer = AdvDiffModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_diff_pruning_modular(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected)

    trainer = ModularDiffModel(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
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
        sparse_task = args_train.sparse_task,
        merged_cutoff = args_train.merged_cutoff,
        merged_min_pct = args_train.merged_min_pct,
        fixmask_pct = args_train.fixmask_pct
    )
    trainer = ModularDiffModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_baseline_task(device, train_loader, val_loader, num_labels, train_logger, args_train):

    loss_fn, pred_fn, metrics = get_callables(num_labels)

    trainer = TaskModel(
        model_name = args_train.model_name,
        num_labels = num_labels,
        dropout = args_train.task_dropout,
        n_hidden = args_train.task_n_hidden,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        num_epochs = args_train.num_epochs,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_head = args_train.learning_rate_task_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        cooldown = args_train.cooldown
    )
    trainer = TaskModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_baseline_adv(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected)

    trainer = AdvModel(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout
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
        num_epochs = args_train.num_epochs,
        num_epochs_warmup = args_train.num_epochs_warmup,
        adv_lambda = args_train.adv_lambda,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_task_head = args_train.learning_rate_task_head,
        learning_rate_adv_head = args_train.learning_rate_adv_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir
    )
    trainer = AdvModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="Whether to run on small subset for testing")
    parser.add_argument("--adv", type=bool, default=False, help="Whether to run adverserial training")
    parser.add_argument("--baseline", type=bool, default=False, help="Set to True if you want to run baseline models (no diff-pruning)")
    parser.add_argument("--modular", type=bool, default=False, help="Whether to run modular training (task only and adverserial)")
    parser.add_argument("--run_adv_attack", type=bool, default=True, help="Set to false if you do not want to run adverserial attack after training")
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--ds", type=str, default="bios", help="dataset")
    base_args = parser.parse_args()

    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    train_cfg = "train_config_baseline" if base_args.baseline else "train_config_diff_pruning"
    data_cfg = f"data_config_{base_args.ds}"
    args_train = argparse.Namespace(**cfg[train_cfg], **cfg[data_cfg], **cfg["model_config"])
    args_attack = argparse.Namespace(**cfg["adv_attack"])

    if base_args.debug:
        args_train = set_num_epochs_debug(args_train)
        args_attack = set_num_epochs_debug(args_attack)
        args_train = set_dir_debug(args_train)

    device = get_device(base_args.gpu_id)

    train_loader, val_loader, num_labels, num_labels_protected = get_data(args_train, ds=base_args.ds, debug=base_args.debug)
    
    train_logger = get_logger(base_args.baseline, base_args.adv, base_args.modular, args_train, base_args.debug)

    print(f"Running {train_logger.logger_name}")

    if base_args.modular:
        trainer = train_diff_pruning_modular(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train)
    elif base_args.baseline and not base_args.adv:
        trainer = train_baseline_task(device, train_loader, val_loader, num_labels, train_logger, args_train)
    elif base_args.baseline and base_args.adv:
        trainer = train_baseline_adv(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train)
    elif not base_args.baseline and not base_args.adv:
        trainer = train_diff_pruning_task(device, train_loader, val_loader, num_labels, train_logger, args_train)
    elif not base_args.baseline and base_args.adv:
        trainer = train_diff_pruning_adv(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train)

    if base_args.run_adv_attack:
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
            cooldown = args_attack.cooldown
        )


if __name__ == "__main__":

    main()

