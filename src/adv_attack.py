import math
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_heads import AdvHead
from src.training_logger import TrainLogger
from src.utils import dict_to_device

from typing import Callable, Dict


@torch.no_grad()
def get_hidden_dataloader(
    trainer: nn.Module,
    loader: DataLoader,
    shuffle: bool = True,
    label_idx: int = 2
):
    trainer.eval()
    hidden_list = []
    label_list = []
    for batch in tqdm(loader, desc="generating embeddings"):
        inputs, labels = batch[0], batch[label_idx]
        inputs = dict_to_device(inputs, trainer.device)
        hidden = trainer._forward(**inputs)
        hidden_list.append(hidden.cpu())
        label_list.append(labels)
    ds = TensorDataset(torch.cat(hidden_list), torch.cat(label_list))
    return DataLoader(ds, shuffle=shuffle, batch_size=loader.batch_size, drop_last=False)


def adv_attack(
    trainer: nn.Module,
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
    lr: float
):

    adv_head = AdvHead(
        adv_count,
        hid_sizes=[trainer.in_size_heads]*(adv_n_hidden+1),
        num_labels=num_labels,
        dropout_prob=adv_dropout
    )
    adv_head.to(trainer.device)

    optimizer = AdamW(adv_head.parameters(), lr=lr)

    if hasattr(trainer, "_remove_parametrizations"):
        trainer._remove_parametrizations()

    train_loader = get_hidden_dataloader(trainer, train_loader)
    val_loader = get_hidden_dataloader(trainer, val_loader, shuffle=False)

    global_step = 0
    train_str = "Epoch {}, {}"
    result_str = lambda x: ", ".join([f"{k}: {v}" for k,v in x.items()])

    train_iterator = trange(num_epochs, desc=train_str.format(0, ""), leave=False, position=0)
    for epoch in train_iterator:

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)

        for step, batch in enumerate(epoch_iterator):

            adv_head.train()

            inputs, labels = batch
            outputs = adv_head(inputs.to(trainer.device))
            loss = trainer._get_mean_loss(outputs, labels.to(trainer.device), loss_fn)

            loss.backward()
            optimizer.step()
            adv_head.zero_grad()

            logger.step_loss(global_step, loss.item(), lr=lr, suffix="adv_attack")

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

            global_step += 1

        adv_head.eval()
        forward_fn = lambda x, adv_head: adv_head(x)
        result = trainer._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics, adv_head=adv_head)

        logger.validation_loss(epoch, result, "adv_attack")

        train_iterator.set_description(
            train_str.format(epoch, result_str(result)), refresh=True
        )

    print("Adv Attack: Final results after " +  train_str.format(epoch, result_str(result)))