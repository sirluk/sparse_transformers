import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from typing import Union, Callable, Dict, Optional

from src.models.model_heads import AdvHead
from src.models.model_diff_modular import ModularDiffModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class ModularDiffModel_2attr(ModularDiffModel):

    def __init__(
        self,
        model_name: str,
        num_labels_task: int,
        num_labels_protected: int,
        num_labels_protected2: int,
        task_dropout: float = .3,
        task_n_hidden: int = 0,
        adv_dropout: float = .3,
        adv_n_hidden: int = 1,
        adv_count: int = 5,
        adv_task_head: bool = True,
        bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        bottleneck_dropout: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            model_name,
            num_labels_task,
            num_labels_protected,
            task_dropout,
            task_n_hidden,
            adv_dropout,
            adv_n_hidden,
            adv_count,
            adv_task_head,
            bottleneck,
            bottleneck_dim,
            bottleneck_dropout,
            **kwargs
        )

        self.num_labels_protected2 = num_labels_protected2
        self.attr_bias_head = AdvHead(adv_count, hid_sizes=[self.in_size_heads]*(adv_n_hidden+1), num_labels=num_labels_protected2, dropout=adv_dropout)


    def forward_bias(self, **x) -> torch.Tensor:
        return self.attr_bias_head(self._forward(**x))


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
        loss_fn_protected2: Callable,
        pred_fn_protected2: Callable,
        metrics_protected2: Dict[str, Callable],
        num_epochs_warmup: int,
        num_epochs_finetune: int,
        num_epochs_fixmask: int,
        alpha_init: Union[int, float],
        concrete_samples: int,
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
        sparse_task: bool,
        merged_cutoff: bool,
        merged_min_pct: float,
        fixmask_pct: Optional[float] = None,
        seed: Optional[int] = None
    ) -> None:

        self.sparse_task = sparse_task

        self.global_step = 0
        num_epochs_finetune += num_epochs_warmup
        num_epochs_total = num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) * num_epochs_finetune
        train_steps_fixmask = len(train_loader) * num_epochs_fixmask

        log_ratio = self.get_log_ratio(concrete_lower, concrete_upper)

        self._init_sparsity_pen(sparsity_pen)

        if not self.parametrized:
            self._add_diff_parametrizations(
                n_parametrizations = 1 + sparse_task,
                p_requires_grad = not sparse_task,
                alpha_init = alpha_init,
                concrete_lower = concrete_lower,
                concrete_upper = concrete_upper,
                structured = structured_diff_pruning
            )
        else:
            assert self.n_parametrizations == 1 + sparse_task, f"model has been trained with sparse_task={sparse_task}"
            assert self.finetune_state or (self.fixmask_state and num_epochs_finetune==0), "model is in fixmask state but num_epochs_fintune>0"

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
                self._finetune_to_fixmask(
                    fixmask_pct,
                    n_parametrizations = (1 + sparse_task),
                    merged_cutoff = merged_cutoff,
                    merged_min_pct = merged_min_pct
                )
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
                loss_fn_protected2,
                _adv_lambda,
                concrete_samples
            )

            result = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics,
                label_idx=1
            )
            logger.validation_loss(epoch, result, "task")

            result_debiased = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics,
                label_idx=1,
                debiased=True
            )
            logger.validation_loss(epoch, result_debiased, suffix="task_debiased")

            result_protected = self.evaluate(
                val_loader,
                loss_fn_protected,
                pred_fn_protected,
                metrics_protected,
                label_idx=2,
                debiased=True
            )
            logger.validation_loss(epoch, result_protected, suffix="protected")

            result_biased_attr = self.evaluate(
                val_loader,
                loss_fn_protected2,
                pred_fn_protected2,
                metrics_protected2,
                label_idx=3,
                debiased=True
            )
            logger.validation_loss(epoch, result_biased_attr, suffix="biased attr")

            # count non zero
            mask_names = ["task", "adv"]
            for i, suffix in enumerate(mask_names[(not sparse_task):]):
                n_p, n_p_zero, n_p_one = self._count_non_zero_params(i)
                n_p_between = n_p - (n_p_zero + n_p_one)
                logger.non_zero_params(epoch, n_p, n_p_zero, n_p_between, suffix)

            result_str = ", " + ", ".join([
                str_suffix(result, "_task"),
                str_suffix(result_debiased, "_task_debiased"),
                str_suffix(result_protected, "_protected"),
                str_suffix(result_biased_attr, "_biased_attr")
            ])
            train_iterator.set_description(
                train_str.format(epoch, self.model_state, result_str), refresh=True
            )

            if self.fixmask_state or ((num_epochs_fixmask == 0) and (epoch >= num_epochs_warmup)):
                cpt = self.save_checkpoint(
                    Path(output_dir),
                    concrete_lower,
                    concrete_upper,
                    structured_diff_pruning,
                    seed
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
        label_idx: int,
        debiased: bool = False
    ) -> dict:
        self.eval()

        if label_idx==1:
            desc = "task"
            forward_fn = lambda x: self(**x)
        elif label_idx==2:
            desc = "protected attribute"
            forward_fn = lambda x: self.forward_protected(**x)
        else:
            desc = "biased attribute",
            forward_fn = lambda x: self.forward_bias(**x)

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
        loss_fn_protected2: Callable,
        adv_lambda: float,
        concrete_samples: int
    ) -> None:

        self.train()

        epoch_str = "training - step {}, loss_biased: {:7.5f}, loss_debiased: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            inputs, labels_task, labels_protected1, labels_protected2 = batch
            inputs = dict_to_device(inputs, self.device)

            concrete_samples = concrete_samples if self.finetune_state else 1
            concrete_samples_task = concrete_samples if self.sparse_task else 1

            ##################################################
            # START STEP TASK
            self.set_debiased(False)

            loss_biased = 0.
            partial_losses_biased = torch.zeros((2,))
            for _ in range(concrete_samples_task):

                outputs = self(**inputs)
                loss_task = loss_fn(outputs, labels_task.to(self.device))
                loss_biased += loss_task

                if self.finetune_state and self.sparse_task:
                    loss_l0 = self._get_sparsity_pen(log_ratio, 0)
                    loss_biased += loss_l0
                else:
                    loss_l0 = torch.tensor(0.)

                partial_losses_biased += torch.tensor([loss_task, loss_l0]).detach()

            loss_biased /= concrete_samples_task
            partial_losses_biased /= concrete_samples_task

            loss_biased.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            self.zero_grad()

            # END STEP TASK
            ##################################################

            ##################################################
            # START STEP DEBIAS
            self.set_debiased(True)

            loss_debiased = 0.
            partial_losses_debiased = torch.zeros((4,))
            for _ in range(concrete_samples):

                hidden = self._forward(**inputs)
                outputs_task = self.task_head(hidden)
                loss_task_adv = loss_fn(outputs_task, labels_task.to(self.device))
                loss_debiased += loss_task_adv

                outputs_protected = self.adv_head.forward_reverse(hidden, lmbda=adv_lambda)
                loss_protected = self._get_mean_loss(outputs_protected, labels_protected1.to(self.device), loss_fn_protected)
                loss_debiased += loss_protected

                outputs_attr_bias = self.attr_bias_head.forward(hidden)
                loss_attr_bias = self._get_mean_loss(outputs_attr_bias, labels_protected2.to(self.device), loss_fn_protected2)
                loss_debiased += loss_attr_bias

                if self.finetune_state:
                    loss_l0_adv = self._get_sparsity_pen(log_ratio, int(self.sparse_task))
                    loss_debiased += loss_l0_adv
                else:
                    loss_l0_adv = torch.tensor(0.)

                partial_losses_debiased += torch.tensor([loss_task_adv, loss_protected, loss_l0_adv, loss_attr_bias]).detach()

            loss_debiased /= concrete_samples
            partial_losses_debiased /= concrete_samples

            loss_debiased.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            self.zero_grad()

            # END STEP DEBIAS
            ##################################################

            # self.scheduler.step()

            losses_dict = {
                "total": loss_biased.item() + loss_debiased.item(),
                "total_biased": loss_biased.item(),
                "task": partial_losses_biased[0],
                "l0": partial_losses_biased[1],
                "total_adv": loss_debiased.item(),
                "task_adv": partial_losses_debiased[0],
                "protected": partial_losses_debiased[1],
                "l0_adv": partial_losses_debiased[2],
                "attr_bias": partial_losses_debiased[3]
            }
            logger.step_loss(self.global_step, losses_dict)

            epoch_iterator.set_description(epoch_str.format(step, loss_biased.item(), loss_debiased.item()), refresh=True)

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

        optimizer_param_groups = [
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
            },
            {
                "params": self.attr_bias_head.parameters(),
                "lr": learning_rate_adv_head
            },
        ]

        optimizer_param_groups.extend(
            self._get_diff_param_groups(learning_rate, weight_decay, learning_rate_alpha)
        )

        if not self.sparse_task:
            optimizer_param_groups.append(
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-9:] == f".original"],
                    "lr": learning_rate,
                    "weight_decay": weight_decay
                }
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
            "num_labels_task": self.num_labels_task,
            "num_labels_protected": self.num_labels_protected,
            "num_labels_protected2": self.num_labels_protected2,
            "task_dropout": self.task_dropout,
            "task_n_hidden": self.task_n_hidden,
            "adv_dropout": self.adv_dropout,
            "adv_n_hidden": self.adv_n_hidden,
            "adv_count": self.adv_count,
            "adv_task_head": self.adv_task_head,
            "bottleneck": self.has_bottleneck,
            "bottleneck_dim": self.bottleneck_dim,
            "bottleneck_dropout": self.bottleneck_dropout,
            "fixmask": self.fixmask_state,
            "encoder_state_dict": self.encoder_module.state_dict(),
            "bottleneck_state_dict": self.bottleneck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict(),
            "attr_bias_head_state_dict": self.attr_bias_head.state_dict(),
            "concrete_lower": concrete_lower,
            "concrete_upper": concrete_upper,
            "structured_diff_pruning": structured_diff_pruning,
            "sparse_task": self.sparse_task
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename_parts = [
            self.model_name.split('/')[-1],
            "modular_" + f"fixmask{self.fixmask_pct}" if self.fixmask_state else "diff_pruning",
            "attr_bias_head",
            "sparse_task" if self.sparse_task else None,
            "merged_head" if not self.adv_task_head else None,
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
        debiased: bool = True,
        map_location: Union[str, torch.device] = torch.device('cpu')
    ) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)

        cls_instance = cls(
            info_dict['model_name'],
            info_dict['num_labels_task'],
            info_dict['num_labels_protected'],
            info_dict['num_labels_protected2'],
            info_dict['task_dropout'],
            info_dict['task_n_hidden'],
            info_dict['adv_dropout'],
            info_dict['adv_n_hidden'],
            info_dict['adv_count'],
            info_dict['adv_task_head'],
            info_dict['bottleneck'],
            info_dict['bottleneck_dim'],
            info_dict['bottleneck_dropout']
        )

        cls_instance._add_diff_parametrizations(
            n_parametrizations = 1 + info_dict["sparse_task"],
            p_requires_grad = not info_dict["sparse_task"],
            fixmask_init = info_dict["fixmask"],
            alpha_init = 5, # standard value for alpha init, not important here as it will be overwritten by checkpoint
            concrete_lower = info_dict['concrete_lower'],
            concrete_upper = info_dict['concrete_upper'],
            structured = info_dict['structured_diff_pruning']
        )

        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.bottleneck.load_state_dict(info_dict['bottleneck_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])
        cls_instance.adv_head.load_state_dict(info_dict['adv_head_state_dict'])
        cls_instance.attr_bias_head.load_state_dict(info_dict['attr_bias_head_state_dict'])

        cls_instance.set_debiased(debiased)

        if remove_parametrizations:
            cls_instance._remove_parametrizations()

        cls_instance.eval()

        return cls_instance

