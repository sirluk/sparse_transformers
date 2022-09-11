import math
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from src.models.model_base import BaseModel
from src.models.model_heads import AdvHead
from src.training_logger import TrainLogger
from src.utils import generate_embeddings

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

    logger.reset()

    adv_head = AdvHead(
        adv_count,
        hid_sizes=[trainer.in_size_heads]*(adv_n_hidden+1),
        num_labels=num_labels,
        dropout_prob=adv_dropout
    )
    adv_head.to(trainer.device)

    optimizer = AdamW(adv_head.parameters(), lr=lr)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs
    # )

    if create_hidden_dataloader:
        train_loader = get_hidden_dataloader(trainer, train_loader, batch_size=batch_size)
        val_loader = get_hidden_dataloader(trainer, val_loader, shuffle=False, batch_size=batch_size)

    global_step = 0
    train_str = "Epoch {}, {}"
    result_str = lambda x: ", ".join([f"{k}: {v}" for k,v in x.items()])

    performance_decrease_counter = 0
    train_iterator = trange(num_epochs, desc=train_str.format(0, ""), leave=False, position=0)
    for epoch in train_iterator:

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)

        for step, (inputs, labels) in enumerate(epoch_iterator):

            adv_head.train()

            outputs = adv_head(inputs.to(trainer.device))
            loss = trainer._get_mean_loss(outputs, labels.to(trainer.device), loss_fn)

            loss.backward()
            optimizer.step()
            # scheduler.step()
            adv_head.zero_grad()

            logger.step_loss(global_step, loss.item(), lr=lr, suffix=logger_suffix)

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

            global_step += 1

        adv_head.eval()
        forward_fn = lambda x, adv_head: adv_head(x)
        result = trainer._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics, adv_head=adv_head)

        logger.validation_loss(epoch, result, logger_suffix)

        train_iterator.set_description(
            train_str.format(epoch, result_str(result)), refresh=True
        )

        if logger.is_best(result["loss"], ascending=True):
            best_result = result
            best_epoch = epoch
            performance_decrease_counter = 0
        else:
            performance_decrease_counter += 1

        if performance_decrease_counter > cooldown:
            break

    print("Adv Attack: Final result after " +  train_str.format(epoch, result_str(result)))
    print("Adv Attack: Best result " +  train_str.format(best_epoch, result_str(best_result)))
    
    return adv_head