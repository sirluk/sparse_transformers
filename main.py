import argparse
import ruamel.yaml as yaml
import torch

from src.models.model_diff_modular import ModularDiffModel
from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.models.model_modular import ModularModel
from src.model_functions import load_cp
from src.adv_attack import adv_attack
from src.data_handler import get_data
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    get_logger,
    get_callables,
    set_optional_args
)


def train_diff_pruning_task(device, train_loader, val_loader, num_labels, train_logger, args_train, encoder_state_dict = None, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)

    trainer = TaskDiffModel(
        model_name = args_train.model_name,
        num_labels = num_labels,
        dropout = args_train.task_dropout,
        n_hidden = args_train.task_n_hidden,
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
        fixmask_pct = args_train.fixmask_pct,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )
    trainer = TaskDiffModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_diff_pruning_adv(device, train_loader, val_loader, num_labels, num_labels_protected, protected_class_weights, train_logger, args_train, encoder_state_dict = None, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected, class_weights = protected_class_weights)

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
    trainer = AdvDiffModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_diff_pruning_modular(device, train_loader, val_loader, num_labels, num_labels_protected, protected_class_weights, train_logger, args_train, encoder_state_dict = None, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected, class_weights = protected_class_weights)

    trainer = ModularDiffModel(
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
    trainer = ModularDiffModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_baseline_task(device, train_loader, val_loader, num_labels, train_logger, args_train, encoder_state_dict = None, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)

    trainer = TaskModel(
        model_name = args_train.model_name,
        num_labels = num_labels,
        dropout = args_train.task_dropout,
        n_hidden = args_train.task_n_hidden,
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
        num_epochs = args_train.num_epochs,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_head = args_train.learning_rate_task_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        cooldown = args_train.cooldown,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )
    trainer = TaskModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_baseline_adv(device, train_loader, val_loader, num_labels, num_labels_protected, protected_class_weights, train_logger, args_train, encoder_state_dict = None, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected, class_weights = protected_class_weights)

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
        num_epochs = args_train.num_epochs,
        num_epochs_warmup = args_train.num_epochs_warmup,
        adv_lambda = args_train.adv_lambda,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_task_head = args_train.learning_rate_task_head,
        learning_rate_adv_head = args_train.learning_rate_adv_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )
    trainer = AdvModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_baseline_modular(device, train_loader, val_loader, num_labels, num_labels_protected, protected_class_weights, train_logger, args_train, encoder_state_dict = None, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected, class_weights = protected_class_weights)

    trainer = ModularModel(
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
        num_epochs_warmup = args_train.num_epochs_warmup,
        num_epochs = args_train.num_epochs,
        adv_lambda = args_train.adv_lambda,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_task_head = args_train.learning_rate_task_head,
        learning_rate_adv_head = args_train.learning_rate_adv_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )
    trainer = ModularModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    parser.add_argument("--adv", action="store_true", help="Whether to run adverserial training")
    parser.add_argument("--baseline", action="store_true", help="Set to True if you want to run baseline models (no diff-pruning)")
    parser.add_argument("--modular", action="store_true", help="Whether to run modular training (task only and adverserial)")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--ds", type=str, default="bios", help="dataset")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    parser.add_argument("--no_adv_attack", action="store_true", help="Set if you do not want to run adverserial attack after training")
    parser.add_argument("--cp_path", type=str, help="Overwrite pre-trained encoder weights")
    parser.add_argument("--cp_is_sd", action="store_true", help="If checkpoint is a state dict")
    parser.add_argument("--cp_model_type", type=str, help="Model type from which to load encoder weights as string (not required if loading state dict directly)")
    parser.add_argument("--cp_modular_biased", action="store_true", help="If loading checkpoint from modular model set debiased state")
    parser.add_argument("--prot_key_idx", type=int, default=0, help="If protected key is type list: index of key to use")
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

    if args_train.weighted_loss_protected and (base_args.modular or base_args.adv):
        train_loader, val_loader, num_labels, num_labels_protected, protected_class_weights = get_data(
            args_train, attr_idx = base_args.prot_key_idx, return_prot_class_weights = True, debug = base_args.debug
        )
        protected_class_weights = torch.tensor(protected_class_weights, device=device[0])
    else:
        train_loader, val_loader, num_labels, num_labels_protected = get_data(
            args_train, attr_idx = base_args.prot_key_idx, return_prot_class_weights = False, debug = base_args.debug
        )
        protected_class_weights = None     
    
    train_logger = get_logger(
        baseline = base_args.baseline,
        adv = base_args.adv,
        modular = base_args.modular,
        args_train = args_train,
        cp_path = (base_args.cp_path is not None),
        prot_key_idx = base_args.prot_key_idx,
        seed = base_args.seed,
        debug = base_args.debug,
    )

    print(f"Running {train_logger.logger_name}")

    if base_args.baseline:
        if base_args.modular:
            trainer = train_baseline_modular(
                device, train_loader, val_loader, num_labels, num_labels_protected, protected_class_weights, train_logger, args_train, encoder_state_dict, base_args.seed
            )
        elif base_args.adv:
            trainer = train_baseline_adv(
                device, train_loader, val_loader, num_labels, num_labels_protected, protected_class_weights, train_logger, args_train, encoder_state_dict, base_args.seed
            )
        else:
            trainer = train_baseline_task(
                device, train_loader, val_loader, num_labels, train_logger, args_train, encoder_state_dict, base_args.seed
            )
    else:
        if base_args.modular:
            trainer = train_diff_pruning_modular(
                device, train_loader, val_loader, num_labels, num_labels_protected, protected_class_weights, train_logger, args_train, encoder_state_dict, base_args.seed
            )
        elif base_args.adv:
            trainer = train_diff_pruning_adv(
                device, train_loader, val_loader, num_labels, num_labels_protected, protected_class_weights, train_logger, args_train, encoder_state_dict, base_args.seed
            )
        else:
            trainer = train_diff_pruning_task(
                device, train_loader, val_loader, num_labels, train_logger, args_train, encoder_state_dict, base_args.seed
            )

    if not base_args.no_adv_attack:
        loss_fn, pred_fn, metrics = get_callables(num_labels_protected, class_weights = protected_class_weights)
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
            logger_suffix = f"adv_attack{'_debiased' if (base_args.adv or base_args.modular) else ''}"
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

