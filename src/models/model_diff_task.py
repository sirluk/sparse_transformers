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

from src.models.model_heads import ClfHead
from src.models.model_base import BasePruningModel, ModelState
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class TaskDiffModel(BasePruningModel):

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = .3,
        n_hidden: int = 0,
        bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        bottleneck_dropout: Optional[float] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        self.num_labels = num_labels
        self.dropout = dropout
        self.n_hidden = n_hidden
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

        self.task_head = ClfHead([self.in_size_heads]*(n_hidden+1), num_labels, dropout=dropout)

    def _forward(self, **x) -> torch.Tensor:
        hidden = super()._forward(**x)
        return self.bottleneck(hidden)

    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        num_epochs_finetune: int,
        num_epochs_fixmask: int,
        alpha_init: Union[int, float],
        concrete_samples: int,
        concrete_lower: float,
        concrete_upper: float,
        structured_diff_pruning: bool,
        sparsity_pen: Union[float,list],
        learning_rate: float,
        learning_rate_bottleneck: float,
        learning_rate_head: float,
        learning_rate_alpha: float,
        optimizer_warmup_steps: int,
        weight_decay: float,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike],
        cooldown: int,
        fixmask_pct: Optional[float] = None,
        seed: Optional[int] = None
    ) -> None:

        self.global_step = 0
        num_epochs_total = num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) * num_epochs_finetune
        train_steps_fixmask = len(train_loader) * num_epochs_fixmask

        log_ratio = self.get_log_ratio(concrete_lower, concrete_upper)

        self._init_sparsity_pen(sparsity_pen)

        if not self.parametrized:
            self._add_diff_parametrizations(
                n_parametrizations = 1,
                p_requires_grad = False,
                alpha_init = alpha_init,
                concrete_lower = concrete_lower,
                concrete_upper = concrete_upper,
                structured = structured_diff_pruning
            )
        else:
            assert self.finetune_state or (self.fixmask_state and num_epochs_finetune==0), "model is in fixmask state but num_epochs_fintune>0"

        self._init_optimizer_and_schedule(
            train_steps_finetune,
            learning_rate,
            learning_rate_head,
            learning_rate_alpha,
            learning_rate_bottleneck,
            weight_decay,
            optimizer_warmup_steps,
        )

        train_str = "Epoch {}, model_state: {}, {}"
        str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k,v in d.items()])

        performance_decrease_counter = 0
        train_iterator = trange(num_epochs_total, desc=train_str.format(0, self.model_state, ""), leave=False, position=0)
        for epoch in train_iterator:

            # switch to fixed mask training
            if epoch == num_epochs_finetune:
                self._finetune_to_fixmask(fixmask_pct)
                self._init_optimizer_and_schedule(
                    train_steps_fixmask,
                    learning_rate,
                    learning_rate_head,
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
                concrete_samples
            )

            result = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics
            )

            logger.validation_loss(epoch, result, "task")

            # count non zero
            n_p, n_p_zero, n_p_one = self._count_non_zero_params()
            n_p_between = n_p - (n_p_zero + n_p_one)
            logger.non_zero_params(epoch, n_p, n_p_zero, n_p_between, "task")

            train_iterator.set_description(
                train_str.format(epoch, self.model_state, str_suffix(result)), refresh=True
            )

            if self.fixmask_state or (num_epochs_fixmask == 0):
                if logger.is_best(result["loss"], ascending=True):
                    cpt = self.save_checkpoint(
                        Path(output_dir),
                        concrete_lower,
                        concrete_upper,
                        structured_diff_pruning,
                        seed
                    )
                    cpt_result = result
                    cpt_epoch = epoch
                    cpt_model_state = self.model_state
                    performance_decrease_counter = 0
                else:
                    performance_decrease_counter += 1

                if performance_decrease_counter>cooldown:
                    break

        print("Final results after " + train_str.format(epoch, self.model_state, str_suffix(result)))
        print("Best result: " + train_str.format(cpt_epoch, cpt_model_state, str_suffix(cpt_result)))

        return cpt


    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
    ) -> dict:
        self.eval()

        forward_fn = lambda x: self(**x)

        return self._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics)


    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        log_ratio: float,
        max_grad_norm: float,
        concrete_samples: int
    ) -> None:
    
        self.train()

        epoch_str = "training - step {}, loss: {:7.5f}, loss l0 pen: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            inputs, labels, _ = batch
            inputs = dict_to_device(inputs, self.device)

            concrete_samples = concrete_samples if self.finetune_state else 1
            
            losses = torch.zeros((3,))
            for _ in range(concrete_samples):

                loss = 0.

                outputs = self(**inputs)
                loss_task = loss_fn(outputs, labels.to(self.device))
                loss += loss_task

                if self.finetune_state:
                    loss_l0 = self._get_sparsity_pen(log_ratio, 0)
                    loss += loss_l0
                else:
                    loss_l0 = torch.tensor(0.)

                losses += torch.tensor([loss, loss_task, loss_l0]).detach()

                loss /= concrete_samples
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            # self.scheduler.step()
            self.zero_grad()

            losses /= concrete_samples
            losses_dict = dict(zip(["total", "task_adv", "l0_adv"], losses.tolist()))
            logger.step_loss(self.global_step, losses_dict)

            epoch_iterator.set_description(epoch_str.format(step, losses[0], losses[2]), refresh=True)

            self.global_step += 1


    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_head: float,
        learning_rate_alpha: float,
        learning_rate_bottleneck: float = 1e-4,
        weight_decay: float = 0.0,
        num_warmup_steps: int = 0
    ) -> None:

        optimizer_param_groups = [
            {
                "params": self.bottleneck.parameters(),
                "lr": learning_rate_bottleneck
            },
            {
                "params": self.task_head.parameters(),
                "lr": learning_rate_head
            }
        ]

        optimizer_param_groups.extend(
            self._get_diff_param_groups(learning_rate, weight_decay, learning_rate_alpha, 0)
        )

        self.optimizer = AdamW(optimizer_param_groups, betas=(0.9, 0.999), eps=1e-08)

        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        # )


    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
        concrete_lower: float,
        concrete_upper: float,
        structured_diff_pruning: bool,
        seed: Optional[int] = None
    ) -> None:
        info_dict = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "dropout": self.dropout,
            "n_hidden": self.n_hidden,
            "bottleneck": self.has_bottleneck,
            "bottleneck_dim": self.bottleneck_dim,
            "bottleneck_dropout": self.bottleneck_dropout,
            "model_state": self.model_state,
            "encoder_state_dict": self.encoder_module.state_dict(),
            "bottleneck_state_dict": self.bottleneck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "concrete_lower": concrete_lower,
            "concrete_upper": concrete_upper,
            "structured_diff_pruning": structured_diff_pruning
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename_parts = [
            self.model_name.split('/')[-1],
            "task_" + f"fixmask{self.fixmask_pct}" if self.fixmask_state else "diff_pruning",
            "cp_init" if self.state_dict_init else None,
            f"seed{seed}" if seed is not None else None
        ]
        filename = "-".join([x for x in filename_parts if x is not None]) + ".pt"
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
            info_dict['num_labels'],
            info_dict['dropout'],
            info_dict['n_hidden'],
            info_dict['bottleneck'],
            info_dict['bottleneck_dim'],
            info_dict['bottleneck_dropout']
        )

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

        if remove_parametrizations:
            cls_instance._remove_parametrizations()

        cls_instance.eval()

        return cls_instance


