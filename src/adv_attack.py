import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from src.models.model_base import BaseModel
from src.models.model_heads import AdvHead
from src.training_logger import TrainLogger
from src.model_functions import train_head
from src.model_functions import generate_embeddings

from typing import Callable, Dict, Optional


@torch.no_grad()
def get_hidden_dataloader(
    trainer: BaseModel,
    loader: DataLoader,
    shuffle: bool = True,
    label_idx: int = 2,
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
    logger_suffix: str = "adv_attack"
):

    adv_head = AdvHead(
        adv_count,
        hid_sizes=[trainer.in_size_heads]*(adv_n_hidden+1),
        num_labels=num_labels,
        dropout=adv_dropout
    )

    if create_hidden_dataloader:
        train_loader = get_hidden_dataloader(trainer, train_loader, batch_size=batch_size)
        val_loader = get_hidden_dataloader(trainer, val_loader, shuffle=False, batch_size=batch_size)

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