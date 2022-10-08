import math
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm, trange
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.models.model_base import BaseModel
from src.models.model_diff_modular import ModularDiffModel
from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device, get_param_from_name

from typing import Optional, Union, Callable, Dict


AVAILABLE_MODEL_CLASSES = [
    TaskModel,
    AdvModel,
    TaskDiffModel,
    AdvDiffModel,
    ModularDiffModel
]


@torch.no_grad()
def merge_models(*model_list) -> torch.nn.Module:
    # assert all weights match
    sets = [set([n for n, _ in m.named_parameters()]) for m in model_list]
    try:
        intersect = sets[0].intersection(*sets[1:])
        assert len(sets[0]) == len(intersect)
    except:
        all_keys = sets[0].union(*sets[1:])
        missing = [k for k in all_keys if k not in intersect]
        raise Exception(f"Keys {missing} not present in all models")

    model_frame = deepcopy(model_list[0])
    for p_name, p in model_frame.named_parameters():
        p.zero_()
        for i in range(len(model_list)):
            p_add = get_param_from_name(model_list[i], p_name)
            p += p_add

    return model_frame


@torch.no_grad()
def merge_adv_models(
    *adv_model_list
):
    diff_weights = [m.get_diff_weights(0, as_module=True) for m in adv_model_list]
    base_weights = adv_model_list[0].get_base_weights(as_module=True)
    for p_name, p in base_weights.named_parameters():
        for dw in diff_weights:
            p_add = get_param_from_name(dw, p_name)
            p += (p_add / len(diff_weights))
    return base_weights


@torch.no_grad()
def merge_diff_models(
    diff_model_list: list,
    base_model: Optional[torch.nn.Module] = None,
    only_first: bool = False
) -> BaseModel:
    model_name = diff_model_list[0].model_name
    model_list = []
    for m in diff_model_list:
        if only_first:
            idx_list = [0]
        else:
            idx_list = list(range(m.n_parametrizations))
        for idx in idx_list:
            model = m.get_diff_weights(idx, as_module=True)
            model = BaseModel(model_name, model.state_dict())
            model_list.append(model)
    # if no base_model is provided take the mean of the weights from the base weights of the diff models
    if base_model is None:
        base_model = BaseModel(model_name)
        bms = [m.get_base_weights() for m in diff_model_list]
        for values in zip(*bms):
            _n, p_list = list(zip(*values))
            _n = _n[0]
            p = get_param_from_name(base_model.encoder, _n)
            p_new = (torch.stack(p_list) / len(p_list)).sum(0)
            p.copy_(p_new)
        model_list.append(base_model)
    else:
        sd = base_model.encoder.state_dict()
        model_list.append(BaseModel(model_name, sd))
    return merge_models(model_list)


@torch.no_grad()
def generate_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    forward_fn: Callable = lambda m, x: m(**x)
):
    device = next(model.parameters()).device
    model.eval()
    emb_list = []
    labels_list = []
    for batch in tqdm(loader, desc="generating embeddings"):
        inputs = batch[0]
        inputs = dict_to_device(inputs, device)
        emb = forward_fn(model, inputs)
        emb_list.append(emb.cpu())
        labels_list.append(batch[1:])
    labels = [torch.cat(x) for x in zip(*labels_list)]
    embeddings = torch.cat(emb_list)
    return embeddings, *labels


def train_head(
    trainer: BaseModel,
    head: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: TrainLogger,
    loss_fn: Callable,
    pred_fn: Callable,
    metrics: Dict[str, Callable],
    optim: Optimizer,
    num_epochs: int,
    lr: float,
    cooldown: int = 5,
    desc: str = ""
):

    logger.reset()

    head.to(trainer.device)

    optimizer = optim(head.parameters(), lr=lr)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs
    # )

    global_step = 0
    train_str = "Epoch {}, {}"
    result_str = lambda x: ", ".join([f"{k}: {v}" for k,v in x.items()])

    performance_decrease_counter = 0
    train_iterator = trange(num_epochs, desc=train_str.format(0, ""), leave=False, position=0)
    for epoch in train_iterator:

        epoch_str = "training {} - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(desc, 0, math.nan), leave=False, position=1)

        for step, (inputs, labels) in enumerate(epoch_iterator):

            head.train()

            outputs = head(inputs.to(trainer.device))
            loss = trainer._get_mean_loss(outputs, labels.to(trainer.device), loss_fn)

            loss.backward()
            optimizer.step()
            # scheduler.step()
            head.zero_grad()

            logger.step_loss(global_step, loss.item(), lr=lr, suffix=desc)

            epoch_iterator.set_description(epoch_str.format(desc, step, loss.item()), refresh=True)

            global_step += 1

        head.eval()
        forward_fn = lambda x, head: head(x)
        result = trainer._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics, head=head)

        logger.validation_loss(epoch, result, desc)

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

    prefix = desc + ': ' if desc else ''
    print(f"{prefix}Final result after " +  train_str.format(epoch, result_str(result)))
    print(f"{prefix}Best result " +  train_str.format(best_epoch, result_str(best_result)))

    return head


def model_factory(
    cp_path: Union[str, Path],
    map_location: Union[str, torch.device] = torch.device('cpu'),
    **kwargs
    ):
    info_dict = torch.load(cp_path, map_location=map_location)
    model_cls = eval(info_dict["cls_name"])
    assert model_cls in AVAILABLE_MODEL_CLASSES, \
        f"Model Class {model_cls} is not in available model classes: {', '.join(AVAILABLE_MODEL_CLASSES)}"
    load_cp_args = model_cls.load_checkpoint.__code__.co_varnames
    model_cls_kwargs = {k:v for k,v in kwargs.items() if k in load_cp_args}
    return model_cls.load_checkpoint(cp_path, map_location=map_location, **model_cls_kwargs)