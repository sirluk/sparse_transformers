import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from src.models.model_base import BaseModel
from src.models.model_heads import AdvHead
from src.training_logger import TrainLogger
from src.model_functions import train_head, generate_embeddings
from src.data_handler import get_data
from src.utils import get_callables

from typing import Callable, Dict, Optional


@torch.no_grad()
def get_hidden_dataloader(
    trainer: BaseModel,
    loader: DataLoader,
    label_idx: int,
    shuffle: bool = True,
    batch_size: Optional[int] = None
):
    data = generate_embeddings(trainer, loader, forward_fn = lambda m, x: m._forward(**x))
    bs = loader.batch_size if batch_size is None else batch_size
    ds = TensorDataset(data[0], data[label_idx])
    return DataLoader(ds, shuffle=shuffle, batch_size=bs, drop_last=False)


def adv_attack(
    trainer: BaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: TrainLogger,
    loss_fn: Callable,
    pred_fn: Callable,
    metrics: Dict[str, Callable],
    num_labels: int,
    adv_n_hidden: int,
    adv_count: int,
    adv_dropout: int,
    num_epochs: int,
    lr: float,
    cooldown: int = 5,
    create_hidden_dataloader: bool = True,
    batch_size: Optional[int] = None,
    label_idx: int = 2,
    logger_suffix: str = "adv_attack"
):

    adv_head = AdvHead(
        adv_count,
        hid_sizes=[trainer.in_size_heads]*(adv_n_hidden+1),
        num_labels=num_labels,
        dropout=adv_dropout
    )

    if create_hidden_dataloader:
        train_loader = get_hidden_dataloader(trainer, train_loader, label_idx=label_idx, batch_size=batch_size)
        val_loader = get_hidden_dataloader(trainer, val_loader, label_idx=label_idx, shuffle=False, batch_size=batch_size)

    adv_head = train_head(
        trainer = trainer,
        head = adv_head,
        train_loader = train_loader,
        val_loader = val_loader,
        logger = logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        optim = AdamW,
        num_epochs = num_epochs,
        lr = lr,
        cooldown = cooldown,
        desc = logger_suffix
    )


def run_adv_attack(
    base_args,
    args_train,
    args_attack,
    trainer,
    train_logger
):

    train_loader, val_loader, _, num_labels_protected, protected_key, protected_class_weights = get_data(
        args_train,
        use_all_attr = True,
        compute_class_weights = args_train.weighted_loss_protected,
        device = trainer.device,
        debug = base_args.debug
    )

    if base_args.modular:
        train_data = []
        val_data = []
        for i in range(trainer.n_embeddings):
            trainer.set_debiased(bool(i), debiased_par_idx=max(0,i-1))
            train_data.append(generate_embeddings(trainer, train_loader, forward_fn = lambda m, x: m._forward(**x)))
            val_data.append(generate_embeddings(trainer, val_loader, forward_fn = lambda m, x: m._forward(**x)))
    else:
        train_data = generate_embeddings(trainer, train_loader, forward_fn = lambda m, x: m._forward(**x))
        val_data = generate_embeddings(trainer, val_loader, forward_fn = lambda m, x: m._forward(**x))


    for (i, (num_lbl_prot, prot_k, prot_w)) in enumerate(zip(num_labels_protected, protected_key, protected_class_weights)):

        label_idx = i+2
        loss_fn, pred_fn, metrics = get_callables(num_lbl_prot, class_weights = prot_w)

        if base_args.modular:
            for emb_idx, (td, tv) in enumerate(zip(train_data, val_data)):
                
                train_loader = DataLoader(
                    TensorDataset(td[0], td[label_idx]), shuffle=True, batch_size=args_attack.attack_batch_size, drop_last=False
                )
                val_loader = DataLoader(
                    TensorDataset(tv[0], tv[label_idx]), shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False
                )

                if emb_idx == 0:
                    emb_name = "task_emb"
                elif base_args.prot_key_idx is None:
                    emb_name = "adv_emb_all" if trainer.adv_merged else f"adv_emb_{protected_key[emb_idx-1]}"
                else:
                    emb_name = f"adv_emb_{protected_key[base_args.prot_key_idx]}"
                
                # if trainer.adv_merged:
                #     debiased_embeddings = (emb_idx == 1) and ((i == base_args.prot_key_idx) or (base_args.prot_key_idx is None))
                # else:
                #     debiased_embeddings = ((i == base_args.prot_key_idx) and (emb_idx == 1)) or ((base_args.prot_key_idx is None) and (i == emb_idx-1))
                # deb_name = f"target_key_{prot_k}" if debiased_embeddings else prot_k
                
                adv_attack(
                    trainer = trainer,
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
                    batch_size = args_attack.attack_batch_size,
                    label_idx = label_idx,
                    logger_suffix = "_".join(["adv_attack", emb_name, f"target_key_{prot_k}"])
                )
        else:

            train_loader = DataLoader(
                TensorDataset(train_data[0], train_data[label_idx]), shuffle=True, batch_size=args_attack.attack_batch_size, drop_last=False
            )
            val_loader = DataLoader(
                TensorDataset(val_data[0], val_data[label_idx]), shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False
            )

            if not base_args.adv:
                emb_name = "task_emb"
            elif base_args.prot_key_idx is None:
                emb_name = f"adv_emb_all"
            else:
                emb_name = f"adv_emb_{protected_key[base_args.prot_key_idx]}"

            # debiased_embeddings = base_args.adv and ((i == base_args.prot_key_idx) or (base_args.prot_key_idx is None))
            # deb_name = f"debiased_{prot_k}" if debiased_embeddings else prot_k

            adv_attack(
                trainer = trainer,
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
                batch_size = args_attack.attack_batch_size,
                label_idx = label_idx,
                logger_suffix = "_".join(["adv_attack", emb_name, f"target_key_{prot_k}"])
            )