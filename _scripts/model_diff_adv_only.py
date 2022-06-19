# TODO: implement bottleneck

import os
import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from typing import Union, Callable, Dict, Optional

from src.models.model_diff_task import TaskDiffModel
from src.models.model_heads import ClfHead, AdvHead
from src.models.model_base import BasePruningModel, ModelState
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class AdvOnlyDiffModel(BasePruningModel):

    def __init__(
        self,
        model_name: str,
        num_labels_task: int,
        num_labels_protected: int,
        task_dropout: float = .3,
        task_n_hidden: int = 0,
        adv_dropout: float = .3,
        adv_n_hidden: int = 1,
        adv_count: int = 5,
        bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        bottleneck_dropout: Optional[float] = None,
        task_diff_checkpoint: Union[str, os.PathLike, None] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)       

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

        if task_diff_checkpoint:
            task_diff_model = TaskDiffModel.load_checkpoint(task_diff_checkpoint, remove_parametrizations=True)
            assert self.model_name == task_diff_model.model_name, "task_diff_checkpoint needs to be same model class"
            self.encoder.load_state_dict(task_diff_model.encoder.state_dict())
            self.encoder_checkpoint = True
        else:
            self.encoder_checkpoint = False

        # bottleneck layer
        if self.has_bottleneck:
            self.bottleneck = ClfHead(self.hidden_size, bottleneck_dim, dropout=dropout)
            self.in_size_heads = bottleneck_dim
        else:
            self.bottleneck = torch.nn.Identity()
            self.in_size_heads = self.hidden_size

        # heads
        self.task_head = ClfHead([self.in_size_heads]*(task_n_hidden+1), num_labels_task, dropout=task_dropout)
        self.adv_head = AdvHead(adv_count, hid_sizes=[self.in_size_heads]*(adv_n_hidden+1), num_labels=num_labels_protected, dropout=adv_dropout)

    def _forward(self, **x) -> torch.Tensor:
        hidden = super()._forward(**x)
        return self.bottleneck(hidden)

    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))

    def forward_protected(self, **x) -> torch.Tensor:
        return self.adv_head(self._forward(**x)) 

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        loss_fn_protected: Callable,
        pred_fn_protected: Callable,
        metrics_protected: Dict[str, Callable],
        num_epochs_warmup: int,
        num_epochs_finetune: int,
        num_epochs_fixmask: int,
        alpha_init: Union[int, float],
        concrete_lower: float,
        concrete_upper: float,
        structured_diff_pruning: bool,
        adv_lambda: float,
        sparsity_pen: Union[float,list],
        fixmask_pct: float,
        learning_rate: float,
        learning_rate_bottleneck: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        learning_rate_alpha: float,
        optimizer_warmup_steps: int,
        weight_decay: float,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike]
    ) -> None:

        self.global_step = 0
        num_epochs_finetune += num_epochs_warmup
        num_epochs_total = num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) * num_epochs_finetune
        train_steps_fixmask = len(train_loader) * num_epochs_fixmask

        log_ratio = self.get_log_ratio(concrete_lower, concrete_upper)

        self._init_sparsity_pen(sparsity_pen)
        self._add_diff_parametrizations(
            n_parametrizations = 1,
            p_requires_grad = False,
            alpha_init = alpha_init,
            concrete_lower = concrete_lower,
            concrete_upper = concrete_upper,
            structured = structured_diff_pruning
        )

        self._init_optimizer_and_schedule(
            train_steps_finetune,
            learning_rate,
            learning_rate_task_head,
            learning_rate_adv_head,
            learning_rate_alpha,
            learning_rate_bottleneck,
            weight_decay,
            optimizer_warmup_steps
        )

        train_str = "Epoch {}, model_state: {}, {}"
        str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k,v in d.items()])

        train_iterator = trange(num_epochs_total, desc=train_str.format(0, self._model_state, ""), leave=False, position=0)
        for epoch in train_iterator:
            if epoch<num_epochs_warmup:
                _adv_lambda = 0.
            else:
                _adv_lambda = adv_lambda

            # switch to fixed mask training
            if epoch == num_epochs_finetune:
                self._finetune_to_fixmask(fixmask_pct)
                self._init_optimizer_and_schedule(
                    train_steps_fixmask,
                    learning_rate,
                    learning_rate_task_head,
                    learning_rate_adv_head,
                    learning_rate_alpha,
                    learning_rate_bottleneck,
                    weight_decay,
                    optimizer_warmup_steps
                )

            self._step(
                train_loader,
                loss_fn,
                logger,
                log_ratio,
                max_grad_norm,
                loss_fn_protected,
                _adv_lambda
            )

            result_protected = self.evaluate(
                val_loader,
                loss_fn_protected,
                pred_fn_protected,
                metrics_protected,
                predict_prot=True
            )
            logger.validation_loss(epoch, result_protected, suffix="protected")

            if epoch<num_epochs_warmup:
                result = {k:None for k in result.keys()}
            else:
                result = self.evaluate(
                    val_loader,
                    loss_fn,
                    pred_fn,
                    metrics
                )
                logger.validation_loss(epoch, result, suffix="task_debiased")

            # count non zero
            n_p, n_p_zero, n_p_between = self._count_non_zero_params()
            logger.non_zero_params(epoch, n_p, n_p_zero, n_p_between, "adv")

            result_str = ", " + ", ".join([
                str_suffix(result, "_task_debiased"),
                str_suffix(result_protected, "_protected")
            ])
            train_iterator.set_description(
                train_str.format(epoch, self._model_state, result_str), refresh=True
            )

            if ((num_epochs_fixmask > 0) and (self._model_state==ModelState.FIXMASK)) or ((num_epochs_fixmask == 0) and (epoch >= num_epochs_warmup)):
                cpt = self.save_checkpoint(
                    Path(output_dir),
                    concrete_lower,
                    concrete_upper,
                    structured_diff_pruning
                )

        print("Final results after " + train_str.format(epoch, self._model_state, result_str))

        return cpt


    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        predict_prot: bool = False
    ) -> dict:
        self.eval()

        if predict_prot:
            desc = "protected attribute"
            label_idx = 2
            forward_fn = lambda x: self.forward_protected(**x)
        else:
            desc = "task"
            label_idx = 1
            forward_fn = lambda x: self(**x)

        return self._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics, label_idx, desc)


    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        log_ratio: float,
        max_grad_norm: float,
        loss_fn_protected: Callable,
        adv_lambda: float
    ) -> None:

        self.train()

        epoch_str = "training - step {}, loss: {:7.5f}, loss without l0 pen: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            inputs, labels_task, labels_protected = batch
            inputs = dict_to_device(inputs, self.device)

            hidden = self._forward(**inputs)
            outputs_task = self.task_head(hidden)
            loss = loss_fn(outputs_task, labels_task.to(self.device))

            outputs_protected = self.adv_head.forward_reverse(hidden, lmbda=adv_lambda)
            if isinstance(outputs_protected, torch.Tensor):
                outputs_protected = [outputs_protected]
            losses_protected = []
            for output in outputs_protected:
                losses_protected.append(loss_fn_protected(output, labels_protected.to(self.device)))
            loss_protected = torch.stack(losses_protected).mean()
            loss += loss_protected

            loss_no_pen = loss.item()

            if self._model_state == ModelState.FINETUNING:
                loss += self._get_sparsity_pen(log_ratio, 0)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            # self.scheduler.step()  # Update learning rate schedule
            self.zero_grad()

            logger.step_loss(self.global_step, loss)

            epoch_iterator.set_description(epoch_str.format(step, loss.item(), loss_no_pen), refresh=True)

            self.global_step += 1


    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        learning_rate_alpha: float,
        learning_rate_bottleneck: float = 1e-4,
        weight_decay: float = 0.0,
        num_warmup_steps: int = 0
    ) -> None:

        optimizer_param_groups = self._get_diff_param_groups(0, learning_rate, weight_decay, learning_rate_alpha)

        optimizer_param_groups.extend([
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
        ])

        self.optimizer = AdamW(optimizer_param_groups, betas=(0.9, 0.999), eps=1e-08)

        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        # )


    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
        concrete_lower: float,
        concrete_upper: float,
        structured_diff_pruning: bool
    ) -> None:
        info_dict = {
            "model_name": self.model_name,
            "encoder_checkpoint": self.encoder_checkpoint,
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
            "model_state": self._model_state,
            "encoder_state_dict": self.encoder_module.state_dict(),
            "bottleneck_state_dict": self.bottleneck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict(),
            "concrete_lower": concrete_lower,
            "concrete_upper": concrete_upper,
            "structured_diff_pruning": structured_diff_pruning
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.model_name.split('/')[-1]}-{'fixmask' if self._model_state == ModelState.FIXMASK else 'diff_pruning'}-adv_only.pt"
        filepath = output_dir / filename
        torch.save(info_dict, filepath)
        return filepath


    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, os.PathLike],
        remove_parametrizations: bool = False,
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

        cls_instance.encoder_checkpoint = info_dict['encoder_checkpoint']

        cls_instance._add_diff_parametrizations(
            n_parametrizations = 1,
            p_requires_grad = False,
            fixmask_init = (info_dict["model_state"] == ModelState.FIXMASK),
            alpha_init = 5, # standard value for alpha init, not important here as it will be overwritten by checkpoint
            concrete_lower = info_dict['concrete_lower'],
            concrete_upper = info_dict['concrete_upper'],
            structured = info_dict['structured_diff_pruning']
        )

        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.bottleneck.load_state_dict(info_dict['bottleneck_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])
        cls_instance.adv_head.load_state_dict(info_dict['adv_head_state_dict'])

        if remove_parametrizations:
            cls_instance._remove_parametrizations()

        cls_instance.eval()

        return cls_instance


