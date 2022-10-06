import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from typing import Union, Callable, Dict, Optional

from src.models.model_heads import ClfHead, AdvHead
from src.models.model_base import BaseModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class AdvModel(BaseModel):

    def __init__(
        self,
        model_name: str,
        num_labels_task: int,
        num_labels_protected: Union[int, list, tuple],
        task_dropout: float = .3,
        task_n_hidden: int = 0,
        adv_dropout: float = .3,
        adv_n_hidden: int = 1,
        adv_count: int = 5,
        bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        bottleneck_dropout: Optional[float] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        if isinstance(num_labels_protected, int):
            num_labels_protected = [num_labels_protected]

        self.num_labels_task = num_labels_task
        self.num_labels_protected = num_labels_protected
        self.task_dropout = task_dropout
        self.task_n_hidden = task_n_hidden
        self.adv_dropout = adv_dropout
        self.adv_n_hidden = adv_n_hidden
        self.adv_count = adv_count
        self.has_bottleneck = bottleneck
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_dropout = bottleneck_dropout

        # bottleneck layer
        if self.has_bottleneck:
            self.bottleneck = ClfHead(self.hidden_size, bottleneck_dim, dropout=bottleneck_dropout)
            self.in_size_heads = bottleneck_dim
        else:
            self.bottleneck = torch.nn.Identity()
            self.in_size_heads = self.hidden_size

        # heads
        self.task_head = ClfHead([self.in_size_heads]*(task_n_hidden+1), num_labels_task, dropout=task_dropout)

        self.adv_head = torch.nn.ModuleList()
        for n in num_labels_protected:
            self.adv_head.append(
                AdvHead(adv_count, hid_sizes=[self.in_size_heads]*(adv_n_hidden+1), num_labels=n, dropout=adv_dropout)
            )

    def _forward(self, **x) -> torch.Tensor:
        hidden = super()._forward(**x)
        return self.bottleneck(hidden)

    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))

    def forward_protected(self, head_idx=0, **x) -> torch.Tensor:
        return self.adv_head[head_idx](self._forward(**x))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        loss_fn_protected: Union[Callable, list, tuple],
        pred_fn_protected: Union[Callable, list, tuple],
        metrics_protected: Union[Dict[str, Callable], list, tuple],
        num_epochs: int,
        num_epochs_warmup: int,
        adv_lambda: float,
        learning_rate: float,
        learning_rate_bottleneck: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        optimizer_warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike],
        protected_key: Optional[Union[str, list, tuple]] = None,
        checkpoint_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:

        if not isinstance(loss_fn_protected, (list, tuple)):
            loss_fn_protected = [loss_fn_protected]
        if not isinstance(pred_fn_protected, (list, tuple)):
            pred_fn_protected = [pred_fn_protected]
        if not isinstance(metrics_protected, (list, tuple)):
            metrics_protected = [metrics_protected]
        if not isinstance(protected_key, (list, tuple)):
            protected_key = [protected_key]

        self.global_step = 0
        num_epochs_total = num_epochs + num_epochs_warmup
        train_steps = len(train_loader) * num_epochs_total

        self._init_optimizer_and_schedule(
            train_steps,
            learning_rate,
            learning_rate_task_head,
            learning_rate_adv_head,
            learning_rate_bottleneck,
            optimizer_warmup_steps
        )

        self.zero_grad()

        train_str = "Epoch {}, {}"
        str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k,v in d.items()])

        train_iterator = trange(num_epochs_total, desc=train_str.format(0, "", ""), leave=False, position=0)
        for epoch in train_iterator:

            if epoch<num_epochs_warmup:
                _adv_lambda = 0.
            else:
                _adv_lambda = adv_lambda

            self._step(
                train_loader,
                loss_fn,
                logger,
                max_grad_norm,
                loss_fn_protected,
                _adv_lambda
            )

            result = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics
            )
            logger.validation_loss(epoch, result, suffix="task_debiased")

            results_protected = []
            for i, (prot_key, loss_fn_prot, pred_fn_prot, metrics_prot) in enumerate(zip(
                protected_key, loss_fn_protected, pred_fn_protected, metrics_protected
            )):
                k = str(prot_key if prot_key is not None else i)
                res_prot = self.evaluate(
                    val_loader,
                    loss_fn_prot,
                    pred_fn_prot,
                    metrics_prot,
                    label_idx=i+1
                )
                results_protected.append((k, res_prot))
                logger.validation_loss(epoch, res_prot, suffix=f"protected_{k}")

            result_strings = [str_suffix(result, "_task_debiased")]
            for (k, r) in results_protected:
                result_strings.append(str_suffix(r, f"_protected_{k}"))
            result_str = ", ".join(result_strings)

            train_iterator.set_description(train_str.format(epoch, result_str), refresh=True)

            cpt = self.save_checkpoint(Path(output_dir), checkpoint_name, seed)

        print("Final result after " + train_str.format(epoch, result_str))

        return cpt


    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        label_idx: int = 0
    ) -> dict:
        self.eval()

        if label_idx > 0:
            desc = f"protected attribute {label_idx-1}"
            forward_fn = lambda x: self.forward_protected(head_idx=label_idx-1, **x)
        else:
            desc = "task"
            forward_fn = lambda x: self(**x)

        return self._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics, label_idx, desc)


    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        max_grad_norm: float,
        loss_fn_protected: Union[Callable, list, tuple],
        adv_lambda: float
    ) -> None:
        self.train()

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            loss = 0.

            inputs, labels_task = batch[:2]
            labels_protected = batch[2:]
            inputs = dict_to_device(inputs, self.device)
            hidden = self._forward(**inputs)
            outputs_task = self.task_head(hidden)
            loss_task = loss_fn(outputs_task, labels_task.to(self.device))
            loss += loss_task

            for i, (l, loss_fn_prot) in enumerate(zip(labels_protected, loss_fn_protected)):
                outputs_protected = self.adv_head[i].forward_reverse(hidden, lmbda = adv_lambda)
                loss_protected = self._get_mean_loss(outputs_protected, l.to(self.device), loss_fn_prot)
                loss += loss_protected

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            # self.scheduler.step()
            self.zero_grad()

            losses_dict = {
                "total_adv": loss.item(),
                "task_adv": loss_task.item(),
                "protected": loss_protected.item()
            }
            logger.step_loss(self.global_step, losses_dict)

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

            self.global_step += 1


    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        learning_rate_bottleneck: float = 1e-4,
        num_warmup_steps: int = 0
    ) -> None:

        optimizer_params = [
            {
                "params": self.encoder.parameters(),
                "lr": learning_rate
            },
            {
                "params": self.bottleneck.parameters(),
                "lr": learning_rate_bottleneck
            },
            {
                "params": self.task_head.parameters(),
                "lr": learning_rate_task_head
            },
            {
                "params": self.adv_head.parameters(),
                "lr": learning_rate_adv_head
            }
        ]

        self.optimizer = AdamW(optimizer_params, betas=(0.9, 0.999), eps=1e-08)

        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        # )


    def make_checkpoint_name(
        self,
        seed: Optional[int] = None
    ):
        filename_parts = [
            self.model_name.split('/')[-1],
            "adv_baseline",
            "cp_init" if self.state_dict_init else None,
            f"seed{seed}" if seed is not None else None
        ]
        return "-".join([x for x in filename_parts if x is not None]) + ".pt"

    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
        checkpoint_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:
        info_dict = {
            "cls_name": self.__class__.__name__,
            "model_name": self.model_name,
            "num_labels_task": self.num_labels_task,
            "num_labels_protected": self.num_labels_protected,
            "task_dropout": self.task_dropout,
            "task_n_hidden": self.task_n_hidden,
            "adv_dropout": self.adv_dropout,
            "adv_n_hidden": self.adv_n_hidden,
            "adv_count": self.adv_count,
            "bottleneck": self.has_bottleneck,
            "bottleneck_dim": self.bottleneck_dim,
            "bottleneck_dropout": self.bottleneck_dropout,
            "encoder_state_dict": self.encoder_module.state_dict(),
            "bottleneck_state_dict": self.bottleneck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict()
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_name is None:
            checkpoint_name = self.make_checkpoint_name(seed)
        filepath = output_dir / checkpoint_name
        torch.save(info_dict, filepath)
        return filepath


    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, os.PathLike],
        map_location: Union[str, torch.device] = torch.device('cpu')
    ) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)

        cls_instance = cls(
            info_dict['model_name'],
            info_dict['num_labels_task'],
            info_dict['num_labels_protected'],
            info_dict['task_dropout'],
            info_dict['task_n_hidden'],
            info_dict['adv_dropout'],
            info_dict['adv_n_hidden'],
            info_dict['adv_count'],
            info_dict['bottleneck'],
            info_dict['bottleneck_dim'],
            info_dict['bottleneck_dropout']
        )

        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.bottleneck.load_state_dict(info_dict['bottleneck_state_dict'])
        cls_instance.adv_head.load_state_dict(info_dict['adv_head_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])

        cls_instance.eval()

        return cls_instance
