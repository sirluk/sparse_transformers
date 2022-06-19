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
from src.models.model_base import BasePruningModel, ModelState
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class ModularDiffModel(BasePruningModel):

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

        # bottleneck layer
        if self.has_bottleneck:
            self.bottleneck_biased = ClfHead(self.hidden_size, bottleneck_dim, dropout=bottleneck_dropout)
            self.bottleneck_debiased = ClfHead(self.hidden_size, bottleneck_dim, dropout=bottleneck_dropout)
            self.in_size_heads = bottleneck_dim
        else:
            self.bottleneck_biased = torch.nn.Identity()
            self.bottleneck_debiased = torch.nn.Identity()
            self.in_size_heads = self.hidden_size

        # heads
        self.task_head_biased = ClfHead([self.in_size_heads]*(task_n_hidden+1), num_labels_task, dropout=task_dropout)
        self.task_head_debiased = ClfHead([self.in_size_heads]*(task_n_hidden+1), num_labels_task, dropout=task_dropout)
        self.adv_head = AdvHead(adv_count, hid_sizes=[self.in_size_heads]*(adv_n_hidden+1), num_labels=num_labels_protected, dropout=adv_dropout)

        self.set_debiased(False)

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
        learning_rate: float,
        learning_rate_bottleneck: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        learning_rate_alpha: float,
        optimizer_warmup_steps: int,
        weight_decay: float,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike],
        fixmask_pct: Optional[float] = None
    ) -> None:

        self.global_step = 0
        num_epochs_finetune += num_epochs_warmup
        num_epochs_total = num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) * num_epochs_finetune
        train_steps_fixmask = len(train_loader) * num_epochs_fixmask

        log_ratio = self.get_log_ratio(concrete_lower, concrete_upper)

        self._init_sparsity_pen(sparsity_pen)
        self._add_diff_parametrizations(
            n_parametrizations = 2,
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

        train_iterator = trange(num_epochs_total, desc=train_str.format(0, self.model_state, ""), leave=False, position=0)
        for epoch in train_iterator:
            if epoch<num_epochs_warmup:
                _adv_lambda = 0.
            else:
                _adv_lambda = adv_lambda

            # switch to fixed mask training
            if epoch == num_epochs_finetune:
                self._finetune_to_fixmask(fixmask_pct, n_parametrizations = 2)
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

            result = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics
            )
            logger.validation_loss(epoch, result, "task")

            result_debiased = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics,
                debiased=True
            )
            logger.validation_loss(epoch, result_debiased, suffix="task_debiased")

            result_protected = self.evaluate(
                val_loader,
                loss_fn_protected,
                pred_fn_protected,
                metrics_protected,
                predict_prot=True
            )
            logger.validation_loss(epoch, result_protected, suffix="protected")

            # count non zero
            for i, suffix in enumerate(["task", "adv"]):
                n_p, n_p_zero, n_p_one = self._count_non_zero_params(i)
                n_p_between = n_p - (n_p_zero + n_p_one)
                logger.non_zero_params(epoch, n_p, n_p_zero, n_p_between, suffix)

            result_str = ", " + ", ".join([
                str_suffix(result, "_task"),
                str_suffix(result_debiased, "_task_debiased"),
                str_suffix(result_protected, "_protected")
            ])
            train_iterator.set_description(
                train_str.format(epoch, self.model_state, result_str), refresh=True
            )

            if ((num_epochs_fixmask > 0) and (self.model_state==ModelState.FIXMASK)) or ((num_epochs_fixmask == 0) and (epoch >= num_epochs_warmup)):
                cpt = self.save_checkpoint(
                    Path(output_dir),
                    concrete_lower,
                    concrete_upper,
                    structured_diff_pruning
                )

        self.set_debiased(True) # make sure debiasing is active at end of training

        print("Final results after " + train_str.format(epoch, self.model_state, result_str))

        return cpt


    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        predict_prot: bool = False,
        debiased: bool = False
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

        if debiased != self._debiased:
            self.set_debiased(debiased, grad_switch=False)
            result = self._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics, label_idx, desc)
            self.set_debiased((not debiased), grad_switch=False)
        else:
            result = self._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics, label_idx, desc)

        return result


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

        epoch_str = "training - step {}, loss_biased: {:7.5f}, loss_debiased: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            inputs, labels_task, labels_protected = batch
            inputs = dict_to_device(inputs, self.device)

            ##################################################
            # START STEP TASK
            self.set_debiased(False)

            outputs = self(**inputs)
            loss = loss_fn(outputs, labels_task.to(self.device))
            loss_task = loss.item()

            if self.model_state == ModelState.FINETUNING:
                loss_l0 = self._get_sparsity_pen(log_ratio, 0)
                loss += loss_l0
            else:
                loss_l0 = torch.tensor(0.)

            loss_biased = loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step() # TODO check if step only at end is better
            self.zero_grad()

            # END STEP TASK
            ##################################################

            ##################################################
            # START STEP DEBIAS
            self.set_debiased(True)

            hidden = self._forward(**inputs)
            outputs_task = self.task_head(hidden)
            loss = loss_fn(outputs_task, labels_task.to(self.device))
            loss_task_adv = loss.item()

            outputs_protected = self.adv_head.forward_reverse(hidden, lmbda=adv_lambda)
            loss_protected = self._get_mean_loss(outputs_protected, labels_protected.to(self.device), loss_fn_protected)
            loss += loss_protected

            if self.model_state == ModelState.FINETUNING:
                loss_l0_adv = self._get_sparsity_pen(log_ratio, 1)
                loss += loss_l0_adv
            else:
                loss_l0_adv = torch.tensor(0.)

            loss_debiased = loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            self.zero_grad()

            # END STEP DEBIAS
            ##################################################

            # self.scheduler.step()

            logger.step_loss(self.global_step, loss.item(), increment_steps=False)
            logger.step_loss(self.global_step, {
                "task": loss_task,
                "l0": loss_l0.item(),
                "task_adv": loss_task_adv,
                "protected": loss_protected.item(),
                "l0_adv": loss_l0_adv.item()
            })

            epoch_iterator.set_description(epoch_str.format(step, loss_biased, loss_debiased), refresh=True)

            self.global_step += 1


    def set_debiased(self, debiased: bool, grad_switch: bool = True) -> None:
        if debiased:
            self.bottleneck = self.bottleneck_debiased
            self.task_head = self.task_head_debiased
            self._activate_parametrizations(True, 1)
            if grad_switch: self._freeze_parametrizations(True, 0)
        else:
            self.bottleneck = self.bottleneck_biased
            self.task_head = self.task_head_biased
            self._activate_parametrizations(False, 1)
            if grad_switch: self._freeze_parametrizations(False, 0)
        self._debiased = debiased


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

        optimizer_param_groups = [
            {
                "params": self.bottleneck_biased.parameters(),
                "lr": learning_rate_bottleneck
            },
            {
                "params": self.bottleneck_debiased.parameters(),
                "lr": learning_rate_bottleneck
            },
            {
                "params": self.task_head_biased.parameters(),
                "lr": learning_rate_task_head
            },
            {
                "params": self.task_head_debiased.parameters(),
                "lr": learning_rate_task_head
            },
            {
                "params": self.adv_head.parameters(),
                "lr": learning_rate_adv_head
            }
        ]

        diff_param_groups_task = self._get_diff_param_groups(learning_rate, weight_decay, learning_rate_alpha, 0)
        diff_param_groups_protected = self._get_diff_param_groups(learning_rate, weight_decay, learning_rate_alpha, 1)
        optimizer_param_groups.extend(diff_param_groups_task)
        optimizer_param_groups.extend(diff_param_groups_protected)

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
            "model_state": self.model_state,
            "encoder_state_dict": self.encoder_module.state_dict(),
            "bottleneck_biased_state_dict": self.bottleneck_biased.state_dict(),
            "bottleneck_debiased_state_dict": self.bottleneck_debiased.state_dict(),
            "task_head_biased_state_dict": self.task_head_biased.state_dict(),
            "task_head_debiased_state_dict": self.task_head_debiased.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict(),
            "concrete_lower": concrete_lower,
            "concrete_upper": concrete_upper,
            "structured_diff_pruning": structured_diff_pruning
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"fixmask{self.fixmask_pct}" if self.model_state == ModelState.FIXMASK else "diff_pruning"
        filename = f"{self.model_name.split('/')[-1]}-{suffix}-modular.pt"
        filepath = output_dir / filename
        torch.save(info_dict, filepath)
        return filepath

    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, os.PathLike],
        remove_parametrizations: bool = False,
        debiased: bool = True,
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

        cls_instance._add_diff_parametrizations(
            n_parametrizations = 2,
            p_requires_grad = False,
            fixmask_init = (info_dict["model_state"] == ModelState.FIXMASK),
            alpha_init = 5, # standard value for alpha init, not important here as it will be overwritten by checkpoint
            concrete_lower = info_dict['concrete_lower'],
            concrete_upper = info_dict['concrete_upper'],
            structured = info_dict['structured_diff_pruning']
        )

        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.bottleneck_biased.load_state_dict(info_dict['bottleneck_biased_state_dict'])
        cls_instance.bottleneck_debiased.load_state_dict(info_dict['bottleneck_debiased_state_dict'])
        cls_instance.task_head_biased.load_state_dict(info_dict['task_head_biased_state_dict'])
        cls_instance.task_head_debiased.load_state_dict(info_dict['task_head_debiased_state_dict'])
        cls_instance.adv_head.load_state_dict(info_dict['adv_head_state_dict'])

        cls_instance.set_debiased(debiased)

        if remove_parametrizations:
            cls_instance._remove_parametrizations()

        cls_instance.eval()

        return cls_instance

