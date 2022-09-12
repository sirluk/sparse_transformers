import math
from pathlib import Path
from functools import reduce
from copy import deepcopy
from tqdm import tqdm, trange
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AutoModel

from src.models.model_base import BaseModel, BasePruningModel
from src.models.model_diff_modular import ModularDiffModel
from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.models.model_modular import ModularModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device

from typing import Optional, Union, Callable, Dict


def merge_models(model_list: list) -> torch.nn.Module:
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
    with torch.no_grad():
        for p_name, p in model_frame.named_parameters():
            p.zero_()
            for i in range(len(model_list)):
                p_add = reduce(lambda a,b: getattr(a,b), [model_list[i]] + p_name.split("."))
                p += p_add

    return model_frame


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
    sd = base_model.encoder.state_dict() if base_model is not None else AutoModel.from_pretrained(model_name)
    model_list.append(BaseModel(model_name, sd))
    return merge_models(model_list)


def load_cp(
    cp_path: Union[str, Path],
    cp_is_sd: bool,
    cp_model_type: Union[str, BaseModel],
    cp_modular_biased: Optional[bool] = None
):
    if cp_path is not None:
        if cp_is_sd:
            return torch.load(cp_path)
        assert cp_model_type is not None, "if cp_path is set cp_model_type needs to be set as well"
        if isinstance(cp_model_type, str):
            model_class = eval(cp_model_type)
        else:
            model_class = cp_model_type
        kwargs = {"filepath": cp_path}
        if isinstance(model_class, BasePruningModel):
            kwargs["remove_parametrizations"] = True
            if isinstance(model_class, ModularDiffModel):
                assert cp_modular_biased is not None, "if model type is Modular, cp_modular_debiased needs to be set"
                kwargs["debiased"] = not cp_modular_biased
        m = model_class.load_checkpoint(**kwargs)
        return m.encoder.state_dict()


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

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)

        for step, (inputs, labels) in enumerate(epoch_iterator):

            head.train()

            outputs = head(inputs.to(trainer.device))
            loss = trainer._get_mean_loss(outputs, labels.to(trainer.device), loss_fn)

            loss.backward()
            optimizer.step()
            # scheduler.step()
            head.zero_grad()

            logger.step_loss(global_step, loss.item(), lr=lr, suffix=desc)

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

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