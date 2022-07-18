import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from typing import Union, List, Tuple, Callable, Dict, Optional

from src.data_handler import (
    get_data_loader_bios,
    get_data_loader_hatespeech,
    get_data_loader_pan16,
    read_label_file
)
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


def get_num_labels(label_file: Union[str, os.PathLike]) -> int:
    num_labels = len(read_label_file(label_file))
    return 1 if num_labels==2 else num_labels


def get_device(gpu: bool, gpu_id: Union[int, list]) -> List[torch.device]:
    if gpu and torch.cuda.is_available():
        if isinstance(gpu_id, int): gpu_id = [gpu_id]
        device = [torch.device(f"cuda:{int(i)}") for i in gpu_id]
    else:
        device = [torch.device("cpu")]
    print(f"Device: {device}")
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


def get_data(args_train: argparse.Namespace, ds: str, debug: bool = False) -> Tuple[DataLoader, DataLoader, int, int]:

    ds_factory = {
        "bios": get_data_loader_bios,
        "pan16": get_data_loader_pan16,
        "hatespeech": get_data_loader_hatespeech
    }
    get_data_loader = ds_factory[ds]

    num_labels = get_num_labels(args_train.labels_task_path)
    num_labels_protected = get_num_labels(args_train.labels_protected_path)
    tokenizer = AutoTokenizer.from_pretrained(args_train.model_name)
    train_loader = get_data_loader(
        tokenizer = tokenizer,
        data_path = args_train.train_pkl,
        labels_task_path = args_train.labels_task_path,
        labels_prot_path = args_train.labels_protected_path,
        batch_size = args_train.batch_size,
        max_length = 200,
        debug = debug
    )
    val_loader = get_data_loader(
        tokenizer = tokenizer,
        data_path = args_train.val_pkl,
        labels_task_path = args_train.labels_task_path,
        labels_prot_path = args_train.labels_protected_path,
        batch_size = args_train.batch_size,
        max_length = 200,
        shuffle = False,
        debug = debug
    )
    return train_loader, val_loader, num_labels, num_labels_protected


def get_name_for_run(
    baseline: bool,
    adv: bool,
    modular: bool,
    args_train: argparse.Namespace,
    debug: bool = False,
    cp_path: Optional[bool] = False,
    seed: Optional[int] = None
):
    run_parts = ["DEBUG" if debug else None]
    if modular:
        run_parts.extend([
            "modular",
            "merged_head" if not args_train.modular_adv_task_head else None
    ])
    else:
        run_parts.append("adverserial" if adv else "task")
    if baseline:
        run_parts.append("baseline")
    else:
        run_parts.append(
            f"diff_pruning_{args_train.fixmask_pct if args_train.num_epochs_fixmask>0 else 'no_fixmask'}"
        )
        if modular:
            run_parts.extend([
                "sparse_task" if args_train.modular_sparse_task else None,
                "merged_cutoff" if args_train.modular_merged_cutoff else None
            ])
    if args_train.bottleneck:
        run_parts.append(f"bottleneck_{args_train.bottleneck_dim}")
    run_parts.extend([
        args_train.model_name.split('/')[-1],
        str(args_train.batch_size),
        str(args_train.learning_rate),
        "cp_init" if cp_path else None,
        f"seed{seed}" if seed is not None else None
    ])
    run_name = "-".join([x for x in run_parts if x is not None])
    return run_name


def get_logger(
    baseline: bool,
    adv: bool,
    modular: bool,
    args_train: argparse.Namespace,
    debug: bool = False,
    cp_path: Optional[bool] = False,
    seed: Optional[int] = None
) -> TrainLogger:

    log_dir = Path(args_train.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger_name = get_name_for_run(baseline, adv, modular, args_train, debug, cp_path, seed)
    return TrainLogger(
        log_dir = log_dir,
        logger_name = logger_name,
        logging_step = args_train.logging_step
    )


def get_callables(num_labels: int) -> Tuple[Callable, Callable, Dict[str, Callable]]:
    if num_labels == 1:
        loss_fn = lambda x, y: torch.nn.BCEWithLogitsLoss()(x.flatten(), y.float())
        pred_fn = lambda x: (x > 0).long()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        pred_fn = lambda x: torch.argmax(x, dim=1)
    metrics = {
        "acc": accuracy,
        "balanced_acc": lambda x, y: accuracy(x, y, balanced=True)
    }
    return loss_fn, pred_fn, metrics