import argparse
import ruamel.yaml as yaml
import torch

from src.models.model_diff_modular_2attr import ModularDiffModel_2attr
from src.models.model_diff_adv_2attr import AdvDiffModel_2attr
from src.models.model_functions import load_cp
from src.adv_attack import adv_attack
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    get_data,
    get_logger,
    get_callables,
    set_optional_args
)


def train_modular(device, train_loader, val_loader, num_labels, num_labels_protected, num_labels_protected2, train_logger, args_train, encoder_state_dict = None, seed = None):

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
        model_state_dict = encoder_state_dict
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
        sparse_task = args_train.modular_sparse_task,
        merged_cutoff = args_train.modular_merged_cutoff,
        merged_min_pct = args_train.modular_merged_min_pct,
        fixmask_pct = args_train.fixmask_pct,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )
    trainer = ModularDiffModel_2attr.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_adv(device, train_loader, val_loader, num_labels, num_labels_protected, num_labels_protected2, train_logger, args_train, encoder_state_dict = None, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected)
    loss_fn_protected2, pred_fn_protected2, metrics_protected2 = get_callables(num_labels_protected2)

    trainer = AdvDiffModel_2attr(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected,
        num_labels_protected2 = num_labels_protected2,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
        model_state_dict = encoder_state_dict
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
        fixmask_pct = args_train.fixmask_pct,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )
    trainer = AdvDiffModel_2attr.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    parser.add_argument("--modular", action="store_true", help="Whether to run modular training (task only and adverserial)")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--ds", type=str, default="bios", help="dataset")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    parser.add_argument("--cp_path", type=str, help="Overwrite pre-trained encoder weights")
    parser.add_argument("--cp_is_sd", action="store_true", help="If checkpoint is a state dict")
    parser.add_argument("--cp_model_type", type=str, help="Model type from which to load encoder weights as string (not required if loading state dict directly)")
    parser.add_argument("--cp_modular_biased", action="store_true", help="If loading checkpoint from modular model set debiased state")
    parser.add_argument("--no_adv_attack", action="store_true", help="Set if you do not want to run adverserial attack after training")
    parser.add_argument("--rev_prot_key_order", action="store_true", help="Reverse order of protected keys")
    parser.add_argument("--debug", action="store_true", help="Whether to run on small subset for testing")
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

    assert isinstance(args_train.protected_key, list) and len(args_train.protected_key)==2, \
        "args_train.protected_key needs to be a list of length 2"
    assert isinstance(args_train.labels_protected_path, list) and len(args_train.labels_protected_path)==2, \
        "args_train.labels_protected_path needs to be a list of length 2"

    if base_args.rev_prot_key_order:
        args_train.protected_key.reverse()
        args_train.labels_protected_path.reverse()

    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    device = get_device(not base_args.cpu, base_args.gpu_id)
    print(f"Device: {device}")

    encoder_state_dict = load_cp(
        cp_path = base_args.cp_path,
        cp_is_sd = base_args.cp_is_sd,
        cp_model_type = base_args.cp_model_type,
        cp_modular_biased = base_args.cp_modular_biased
    )

    train_loader, val_loader, num_labels, num_labels_protected, num_labels_protected2 = \
        get_data(args_train, use_all_attr=True, debug=base_args.debug)
    
    train_logger = get_logger(
        baseline = False,
        adv = (not base_args.modular),
        modular = base_args.modular,
        args_train = args_train,
        cp_path = (base_args.cp_path is not None),
        prot_key_idx = 0,
        seed = base_args.seed,
        debug = base_args.debug,
        suffix = "2attr"
    )

    print(f"Running {train_logger.logger_name}")

    if base_args.modular:
        trainer = train_modular(device, train_loader, val_loader, num_labels, num_labels_protected, num_labels_protected2, train_logger, args_train, encoder_state_dict, base_args.seed)
    else:
        trainer = train_adv(device, train_loader, val_loader, num_labels, num_labels_protected, num_labels_protected2, train_logger, args_train, encoder_state_dict, base_args.seed)

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
            cooldown = args_attack.cooldown,
            batch_size = args_attack.attack_batch_size,
            logger_suffix = f"adv_attack_debiased"
        )
        if base_args.modular:
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
                cooldown = args_attack.cooldown,
                batch_size = args_attack.attack_batch_size,
                logger_suffix = "adv_attack"
            )


if __name__ == "__main__":

    main()

