from pathlib import Path
import argparse
import ruamel.yaml as yaml
import torch

from src.models.model_doublediff import DoubleDiffModel
from src.adv_attack import adv_attack
from src.training_logger import TrainLogger
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    get_data,
    get_callables,
    set_optional_args
)

torch.manual_seed(0)


def train_doublediff_pruning(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected)

    trainer = DoubleDiffModel(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected,
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
        fixmask_pct = args_train.fixmask_pct,
        seed = seed
    )
    trainer = DoubleDiffModel.load_checkpoint(trainer_cp)
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

    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

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

    device = get_device(not base_args.cpu, base_args.gpu_id)

    train_loader, val_loader, num_labels, num_labels_protected = get_data(args_train, ds=base_args.ds, debug=base_args.debug)

    log_dir = Path(args_train.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger_name = "-".join([x for x in [
        "DEBUG" if base_args.debug else "",
        f"doublediff_{args_train.fixmask_pct if args_train.num_epochs_fixmask>0 else 'no_fixmask'}",
        f"bottleneck_{args_train.bottleneck_dim}" if args_train.bottleneck else "",
        args_train.model_name.split('/')[-1],
        str(args_train.batch_size),
        str(args_train.learning_rate),
        f"seed{base_args.seed}"
    ] if len(x)>0])
    train_logger = TrainLogger(
        log_dir = log_dir,
        logger_name = logger_name,
        logging_step = args_train.logging_step
    )

    print(f"Running {train_logger.logger_name}")

    trainer = train_doublediff_pruning(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train)

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
            cooldown = args_attack.cooldown
        )


if __name__ == "__main__":

    main()

