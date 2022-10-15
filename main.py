import argparse
import ruamel.yaml as yaml
import torch

from src.models.model_diff_modular import ModularDiffModel
from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.model_functions import model_factory
from src.adv_attack import run_adv_attack
from src.data_handler import get_data
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    get_logger,
    get_callables_wrapper,
    set_optional_args
)


def train_diff_pruning_task(
    device,
    train_loader,
    val_loader,
    num_labels,
    loss_fn,
    pred_fn,
    metrics,
    train_logger,
    args_train,
    encoder_state_dict = None,
    cp_load_to_par = False,
    seed = None
):

    trainer = TaskDiffModel(
        model_name = args_train.model_name,
        num_labels = num_labels,
        dropout = args_train.task_dropout,
        n_hidden = args_train.task_n_hidden,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
        concrete_lower = args_train.concrete_lower,
        concrete_upper = args_train.concrete_upper,
        structured_diff_pruning = args_train.structured_diff_pruning,
        alpha_init = args_train.alpha_init,
        encoder_state_dict = encoder_state_dict,
        state_dict_load_to_par = cp_load_to_par
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
        concrete_samples = args_train.concrete_samples,
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


def train_diff_pruning_adv(
    device,
    train_loader,
    val_loader,
    num_labels,
    num_labels_protected_list,
    protected_key_list,
    loss_fn,
    pred_fn,
    metrics,
    loss_fn_protected_list,
    pred_fn_protected_list,
    metrics_protected_list,
    train_logger,
    args_train,
    encoder_state_dict = None,
    cp_load_to_par = False,
    seed = None
):

    trainer = AdvDiffModel(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected_list,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
        concrete_lower = args_train.concrete_lower,
        concrete_upper = args_train.concrete_upper,
        structured_diff_pruning = args_train.structured_diff_pruning,
        alpha_init = args_train.alpha_init,
        encoder_state_dict = encoder_state_dict,
        state_dict_load_to_par = cp_load_to_par
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        loss_fn_protected = loss_fn_protected_list,
        pred_fn_protected = pred_fn_protected_list,
        metrics_protected = metrics_protected_list,
        num_epochs_warmup = args_train.num_epochs_warmup,
        num_epochs_finetune = args_train.num_epochs_finetune,
        num_epochs_fixmask = args_train.num_epochs_fixmask,
        concrete_samples = args_train.concrete_samples,
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
        protected_key = protected_key_list,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )
    trainer = AdvDiffModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_diff_pruning_modular(
    device,
    train_loader, 
    val_loader,
    num_labels,
    num_labels_protected_list,
    protected_key_list,
    loss_fn,
    pred_fn,
    metrics,
    loss_fn_protected_list,
    pred_fn_protected_list,
    metrics_protected_list,
    train_logger,
    args_train,
    encoder_state_dict = None,
    seed = None
):

    trainer = ModularDiffModel(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected_list,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
        concrete_lower = args_train.concrete_lower,
        concrete_upper = args_train.concrete_upper,
        structured_diff_pruning = args_train.structured_diff_pruning,
        alpha_init = args_train.alpha_init,
        adv_task_head = args_train.modular_adv_task_head,
        freeze_single_task_head = args_train.modular_freeze_single_task_head,
        adv_merged = args_train.modular_adv_merged,
        sparse_task = args_train.modular_sparse_task,
        encoder_state_dict = encoder_state_dict
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        loss_fn_protected = loss_fn_protected_list,
        pred_fn_protected = pred_fn_protected_list,
        metrics_protected = metrics_protected_list,
        num_epochs_warmup = args_train.num_epochs_warmup,
        num_epochs_finetune = args_train.num_epochs_finetune,
        num_epochs_fixmask = args_train.num_epochs_fixmask,
        concrete_samples = args_train.concrete_samples,
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
        merged_cutoff = args_train.modular_merged_cutoff,
        merged_min_pct = args_train.modular_merged_min_pct,
        cooldown = args_train.cooldown,
        fixmask_pct = args_train.fixmask_pct,
        protected_key = protected_key_list,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )
    trainer = ModularDiffModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_baseline_task(
    device,
    train_loader,
    val_loader,
    num_labels,
    loss_fn,
    pred_fn,
    metrics,
    train_logger,
    args_train,
    encoder_state_dict = None,
    seed = None
):

    trainer = TaskModel(
        model_name = args_train.model_name,
        num_labels = num_labels,
        dropout = args_train.task_dropout,
        n_hidden = args_train.task_n_hidden,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
        encoder_state_dict = encoder_state_dict
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


def train_baseline_adv(
    device,
    train_loader,
    val_loader,
    num_labels,
    num_labels_protected_list,
    protected_key_list,
    loss_fn,
    pred_fn,
    metrics,
    loss_fn_protected_list,
    pred_fn_protected_list,
    metrics_protected_list,
    train_logger,
    args_train,
    encoder_state_dict = None,
    seed = None
):

    trainer = AdvModel(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected_list,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout,
        encoder_state_dict = encoder_state_dict
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        loss_fn_protected = loss_fn_protected_list,
        pred_fn_protected = pred_fn_protected_list,
        metrics_protected = metrics_protected_list,
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
        protected_key = protected_key_list,
        checkpoint_name = train_logger.logger_name + ".pt",
        seed = seed
    )
    trainer = AdvModel.load_checkpoint(trainer_cp)
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
    parser.add_argument("--cp_modular_biased", action="store_true", help="If loading checkpoint from modular model set debiased state")
    parser.add_argument("--cp_load_to_par", action="store_true", help="initialize checkpoint weights in parametrizations (doesent work for modular model)")
    parser.add_argument("--prot_key_idx", type=int, help="If protected key is type list: index of key to use, if none use all available attributes for taining")
    parser.add_argument("--debug", action="store_true", help="Whether to run on small subset for testing")
    parser.add_argument("--logger_suffix", type=str, help="Add addtional string to logger name")
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

    print(f"args_train:\n{args_train}")

    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    device = get_device(not base_args.cpu, base_args.gpu_id)
    print(f"Device: {device}")

    if base_args.cp_path is not None:
        encoder_state_dict = model_factory(
            cp_path = base_args.cp_path,
            remove_parametrizations = True,
            debiased = (not base_args.cp_modular_biased)
        ).encoder.state_dict()
    else:
        encoder_state_dict = None

    train_loader, val_loader, num_labels, num_labels_protected_list, protected_key_list, protected_class_weights_list = get_data(
        args_train = args_train,
        use_all_attr = (base_args.prot_key_idx is None),
        attr_idx_prot = base_args.prot_key_idx,
        compute_class_weights = args_train.weighted_loss_protected,
        device = device[0],
        debug = base_args.debug
    )

    loss_fn, pred_fn, metrics, loss_fn_protected_list, pred_fn_protected_list, metrics_protected_list = get_callables_wrapper(
        num_labels = num_labels,
        num_labels_protected = num_labels_protected_list,
        protected_class_weights = protected_class_weights_list
    )
    
    train_logger = get_logger(
        baseline = base_args.baseline,
        adv = base_args.adv,
        modular = base_args.modular,
        args_train = args_train,
        cp_path = (base_args.cp_path is not None),
        prot_key_idx = base_args.prot_key_idx,
        seed = base_args.seed,
        debug = base_args.debug,
        suffix = base_args.logger_suffix
    )

    print(f"Running {train_logger.logger_name}")

    if base_args.modular:
        trainer = train_diff_pruning_modular(
            device,
            train_loader,
            val_loader,
            num_labels,
            num_labels_protected_list,
            protected_key_list,
            loss_fn,
            pred_fn,
            metrics,
            loss_fn_protected_list,
            pred_fn_protected_list,
            metrics_protected_list,
            train_logger,
            args_train,
            encoder_state_dict,
            base_args.seed
        )
    elif base_args.baseline:
        if base_args.adv:
            trainer = train_baseline_adv(
                device,
                train_loader,
                val_loader,
                num_labels,
                num_labels_protected_list,
                protected_key_list,
                loss_fn,
                pred_fn,
                metrics,
                loss_fn_protected_list,
                pred_fn_protected_list,
                metrics_protected_list,
                train_logger,
                args_train,
                encoder_state_dict,
                base_args.seed
            )
        else:
            trainer = train_baseline_task(
                device,
                train_loader,
                val_loader,
                num_labels,
                loss_fn,
                pred_fn,
                metrics,
                train_logger,
                args_train,
                encoder_state_dict,
                base_args.seed
            )
    else:
        if base_args.adv:
            trainer = train_diff_pruning_adv(
                device,
                train_loader,
                val_loader,
                num_labels,
                num_labels_protected_list,
                protected_key_list,
                loss_fn,
                pred_fn,
                metrics,
                loss_fn_protected_list,
                pred_fn_protected_list,
                metrics_protected_list,
                train_logger,
                args_train,
                encoder_state_dict,
                base_args.cp_load_to_par,
                base_args.seed
            )
        else:
            trainer = train_diff_pruning_task(
                device,
                train_loader,
                val_loader,
                num_labels,
                loss_fn,
                pred_fn,
                metrics,
                train_logger,
                args_train,
                encoder_state_dict,
                base_args.cp_load_to_par,
                base_args.seed
            )

    if not base_args.no_adv_attack:

        run_adv_attack(
            base_args,
            args_train,
            args_attack,
            trainer,
            train_logger
        )


if __name__ == "__main__":

    main()

