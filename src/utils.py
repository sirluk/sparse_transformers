import argparse
from pathlib import Path
import torch

from typing import Union, List, Tuple, Callable, Dict, Optional

from src.training_logger import TrainLogger
from src.metrics import accuracy


def concrete_stretched(
    alpha: torch.Tensor,
    l: Union[float, int] = -1.5,
    r: Union[float, int] = 1.5,
    deterministic: bool = False
) -> torch.Tensor:
    if not deterministic:
        u = torch.zeros_like(alpha).uniform_().clamp_(0.0001, 0.9999)
        u_term = u.log() - (1-u).log()
    else:
        u_term = 0.
    s = (torch.sigmoid(u_term + alpha))
    s_stretched = s*(r-l) + l
    z = s_stretched.clamp(0, 1000).clamp(-1000, 1)
    return z


def dict_to_device(d: dict, device: Union[str, torch.device]) -> dict:
    return {k:v.to(device) for k,v in d.items()}


def get_device(gpu: bool, gpu_id: Union[int, list]) -> List[torch.device]:
    if gpu and torch.cuda.is_available():
        if isinstance(gpu_id, int): gpu_id = [gpu_id]
        device = [torch.device(f"cuda:{int(i)}") for i in gpu_id]
    else:
        device = [torch.device("cpu")]
    return device


def set_num_epochs_debug(args_obj: argparse.Namespace, num: int = 1) -> argparse.Namespace:
    epoch_args = [n for n in dir(args_obj) if n[:10]=="num_epochs"]
    for epoch_arg in epoch_args:
        v = min(getattr(args_obj, epoch_arg), num)
        setattr(args_obj, epoch_arg, v)
    return args_obj


def set_dir_debug(args_obj: argparse.Namespace) -> argparse.Namespace:
    dir_list = ["output_dir", "log_dir"]
    for d in dir_list:
        v = getattr(args_obj, d)
        setattr(args_obj, d, f"DEBUG_{v}")
    return args_obj


def get_name_for_run(
    baseline: bool,
    adv: bool,
    modular: bool,
    args_train: argparse.Namespace,
    cp_path: bool = False,
    prot_key_idx: int = 0,
    seed: Optional[int] = None,
    debug: bool = False,
    suffix: Optional[str] = None
):
    run_parts = ["DEBUG" if debug else None]

    if modular:
        run_parts.extend([
            "modular",
            "merged_head" if not args_train.modular_adv_task_head else None
    ])
    elif adv:
        run_parts.append("adverserial")
    else:
        run_parts.append("task")

    if baseline:
        run_parts.append("baseline")
    else:
        run_parts.extend([
            f"diff_pruning_{args_train.fixmask_pct if args_train.num_epochs_fixmask>0 else 'no_fixmask'}",
            f"a_samples_{args_train.concrete_samples}" if args_train.concrete_samples > 1 else None
        ])
        if modular:
            run_parts.extend([
                "sparse_task" if args_train.modular_sparse_task else None,
                "merged_cutoff" if args_train.modular_merged_cutoff else None
            ])

    prot_attr = args_train.protected_key if isinstance(args_train.protected_key, str) else args_train.protected_key[prot_key_idx]

    run_parts.extend([
        f"bottleneck_{args_train.bottleneck_dim}" if args_train.bottleneck else None,
        args_train.model_name.split('/')[-1],
        str(args_train.batch_size),
        str(args_train.learning_rate),
        "cp_init" if cp_path else None,
        "weighted_loss_prot" if args_train.weighted_loss_protected and (adv or modular) else None,
        prot_attr if (adv or modular) else None,
        f"seed{seed}" if seed is not None else None,
        suffix,
    ])
    run_name = "-".join([x for x in run_parts if x is not None])
    return run_name


def get_logger(
    baseline: bool,
    adv: bool,
    modular: bool,
    args_train: argparse.Namespace,
    cp_path: bool = False,
    prot_key_idx: int = 0,
    seed: Optional[int] = None,
    debug: bool = False,
    suffix: Optional[str] = None
) -> TrainLogger:

    log_dir = Path(args_train.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger_name = get_name_for_run(baseline, adv, modular, args_train, cp_path, prot_key_idx, seed, debug, suffix)
    return TrainLogger(
        log_dir = log_dir,
        logger_name = logger_name,
        logging_step = args_train.logging_step
    )


def get_logger_custom(
    log_dir: Union[str, Path],
    logger_name: str,
    logging_step: int = 1
) -> TrainLogger:

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return TrainLogger(
        log_dir = log_dir,
        logger_name = logger_name,
        logging_step = logging_step
    )


def get_callables(num_labels: int, class_weights: Optional[Union[int, float, list, torch.tensor]] = None) -> Tuple[Callable, Callable, Dict[str, Callable]]:

    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights)
        if class_weights.dim() == 0:
            class_weights = class_weights.unsqueeze(0)
        if num_labels == 1:
            class_weights = class_weights[1] if len(class_weights)==2 else class_weights[0]

    if num_labels == 1:
        loss_fn = lambda x, y: torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)(x.flatten(), y.float())
        pred_fn = lambda x: (x > 0).long()
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        pred_fn = lambda x: torch.argmax(x, dim=1)
    metrics = {
        "acc": accuracy,
        "balanced_acc": lambda x, y: accuracy(x, y, balanced=True)
    }
    return loss_fn, pred_fn, metrics


def set_optional_args(args_obj: argparse.Namespace, optional_args: list) -> argparse.Namespace:
    ignored = []
    for arg in optional_args:
        assert arg.startswith("--"), "arguments need to start with '--'"
        arg_name = arg.split("=")[0][2:]
        if arg_name in args_obj:
            arg_dtype = type(getattr(args_obj, arg_name))
            if "=" in arg:
                v = arg.split("=")[1]
                arg_value = arg_dtype(v) if arg_dtype!=bool else eval(v)
            else:
                arg_value = True
            setattr(args_obj, arg_name, arg_value)
        else:
            ignored.append(arg)

    if len(ignored) > 0: print(f"ignored args: {ignored}")

    return args_obj