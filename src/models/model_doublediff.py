import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.utils.parametrize as parametrize
from transformers import get_linear_schedule_with_warmup

from typing import Union, Callable, Dict, Optional, List, Tuple

from src.models.model_heads import ClfHead, AdvHead
from src.models.model_base import BasePruningModel, ModelState
from src.models.weight_parametrizations import DoubleDiffWeightFinetune, DiffWeightFixmask
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class DoubleDiffModel(BasePruningModel):

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
        fixmask_pct: Optional[float] = None,
        seed: Optional[int] = None
    ) -> None:

        self.global_step = 0
        num_epochs_finetune += num_epochs_warmup
        num_epochs_total = num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) * num_epochs_finetune
        train_steps_fixmask = len(train_loader) * num_epochs_fixmask

        log_ratio = self.get_log_ratio(concrete_lower, concrete_upper)

        self._init_sparsity_pen(sparsity_pen)
        self._add_diff_parametrizations(
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
                _adv_lambda,
                concrete_samples
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
                predict_prot=True,
                debiased=True
            )
            logger.validation_loss(epoch, result_protected, suffix="protected")

            # count non zero
            n_p, n_p_zero, n_p_between = self._count_non_zero_params()
            for i, suffix in enumerate(["task", "adv"]):
                logger.non_zero_params(epoch, n_p[i], n_p_zero[i], n_p_between[i], suffix)

            result_str = ", " + ", ".join([
                str_suffix(result, "_task"),
                str_suffix(result_debiased, "_task_debiased"),
                str_suffix(result_protected, "_protected")
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
            self.set_debiased(debiased)
            result = self._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics, label_idx, desc)
            self.set_debiased(not debiased)
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
        adv_lambda: float,
        concrete_samples: int
    ) -> None:

        self.train()

        epoch_str = "training - step {}, loss_biased: {:7.5f}, loss_debiased: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            inputs, labels_task, labels_protected = batch
            inputs = dict_to_device(inputs, self.device)

            concrete_samples = concrete_samples if self.finetune_state else 1

            ##################################################
            # START STEP TASK
            self.set_debiased(False)

            losses_biased = torch.zeros((3,))
            for _ in range(concrete_samples):

                loss = 0.

                outputs = self(**inputs)
                loss_task = loss_fn(outputs, labels_task.to(self.device))
                loss += loss_task

                if self.finetune_state:
                    loss_l0 = self._get_sparsity_pen(log_ratio)
                    loss += loss_l0
                else:
                    loss_l0 = torch.tensor(0.)

                losses_biased += torch.tensor([loss, loss_task, loss_l0]).detach()

                loss /= concrete_samples
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step() # TODO check if step only at end is better
            self.zero_grad()
            losses_biased /= concrete_samples

            # END STEP TASK
            ##################################################

            ##################################################
            # START STEP DEBIAS
            self.set_debiased(True)

            losses_debiased = torch.zeros((4,))
            for _ in range(concrete_samples):

                loss = 0.

                hidden = self._forward(**inputs)
                outputs_task = self.task_head(hidden)
                loss = loss_fn(outputs_task, labels_task.to(self.device))
                loss_task_adv = loss.item()

                outputs_protected = self.adv_head.forward_reverse(hidden, lmbda=adv_lambda)
                loss_protected = self._get_mean_loss(outputs_protected, labels_protected.to(self.device), loss_fn_protected)
                loss += loss_protected

                if self.finetune_state:
                    loss_l0_adv = self._get_sparsity_pen(log_ratio)
                    loss += loss_l0_adv
                else:
                    loss_l0_adv = torch.tensor(0.)

                losses_debiased += torch.tensor([loss, loss_task_adv, loss_protected, loss_l0_adv]).detach()

                loss /= concrete_samples
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            self.zero_grad()
            losses_debiased /= concrete_samples

            # END STEP DEBIAS
            ##################################################

            # self.scheduler.step()

            losses_dict = {
                "total": losses_biased[0],
                "task": losses_biased[1],
                "l0": losses_biased[2],
                "total_adv": losses_debiased[0],
                "task_adv": losses_debiased[1],
                "protected": losses_debiased[2],
                "l0_adv": losses_debiased[3]
            }
            logger.step_loss(self.global_step, losses_dict)

            epoch_iterator.set_description(epoch_str.format(step, losses_biased[0], losses_debiased[0]), refresh=True)

            self.global_step += 1


    def set_debiased(self, debiased: bool) -> None:
        try:
            check = (debiased != self._debiased)
        except AttributeError:
            check = True
        if check:
            if debiased:
                self.bottleneck = self.bottleneck_debiased
                self.task_head = self.task_head_debiased
            else:
                self.bottleneck = self.bottleneck_biased
                self.task_head = self.task_head_biased

            if self.fixmask_state:
                self._activate_parametrizations(debiased, 1)
                self._freeze_parametrizations(debiased, 0)
            elif self.finetune_state:
                for base_module in self.get_encoder_base_modules():
                    for par_list in base_module.parametrizations.values():
                        par_list[0].set_weight_state(first = not debiased)
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

        optimizer_param_groups = self._get_diff_param_groups(learning_rate, weight_decay, learning_rate_alpha)

        optimizer_param_groups.extend([
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
        structured_diff_pruning: bool,
        seed: Optional[int] = None
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
        
        suffix = f"fixmask{self.fixmask_pct}" if self.fixmask_state else "diff_pruning"
        seed_str = f"-seed{seed}" if seed is not None else ""
        filename = f"{self.model_name.split('/')[-1]}-doublediff_{suffix}{seed_str}.pt"
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


    @torch.no_grad()
    def _count_non_zero_params(self) -> Tuple[int, int]:
        assert self.parametrized, "Function only implemented for diff pruning"

        n_p = [0, 0]
        n_p_zero = [0, 0]
        n_p_one = [0, 0]
        with self.deterministic():
            for base_module in self.get_encoder_base_modules():
                for n, par_list in list(base_module.parametrizations.items()):
                    if isinstance(par_list[0], DiffWeightFixmask):
                        for idx, par in enumerate(par_list):
                            n_p_ = par_list[idx].mask.numel()
                            n_p_zero_ = (~par_list[idx].mask).sum().item()
                            n_p[idx] += n_p_
                            n_p_zero[idx] += n_p_zero_
                            n_p_one[idx] += (n_p_ - n_p_zero_)
                    else:
                        z_masks = par_list[0].get_z_masks()
                        for idx, z in enumerate(z_masks):
                            n_p[idx] += z.numel()
                            n_p_zero[idx] += (z == 0.).sum().item()
                            n_p_one[idx] += (z == 1.).sum().item()

        n_p_between = [n_p_ - (n_p_zero_ + n_p_one_) for n_p_, n_p_zero_, n_p_one_ in zip(n_p, n_p_zero, n_p_one)]
        return n_p, n_p_zero, n_p_between


    def _add_diff_parametrizations(self, fixmask_init: bool = False, **kwargs) -> None:
        assert not self.parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
        for base_module in self.get_encoder_base_modules(): 
            for n,p in list(base_module.named_parameters()):
                p.requires_grad = False
                if fixmask_init:
                    for _ in range(2):
                        # in case of fixmask init, can only initalize with dummy values
                        parametrize.register_parametrization(base_module, n, DiffWeightFixmask(
                            torch.zeros_like(p), torch.ones_like(p, dtype=bool)
                        ))
                else:
                    parametrize.register_parametrization(base_module, n, DoubleDiffWeightFinetune(p, **kwargs))
        if fixmask_init:
            self.model_state = ModelState.FIXMASK
        else:
            self.model_state = ModelState.FINETUNING


    @torch.no_grad()
    def _finetune_to_fixmask(self, pct: Optional[List[float]] = None) -> None:

        def _get_cutoff(values, pct):
            k = int(len(values) * pct)
            return torch.topk(torch.abs(values), k, largest=True, sorted=True)[0][-1]

        assert self.model_state == ModelState.FINETUNING, "model needs to be in finetuning state"

        with self.deterministic():

            if pct is not None:

                diff_weights = [torch.tensor([])] * 2
                for base_module in self.get_encoder_base_modules():
                    for n, par_list in list(base_module.parametrizations.items()):
                        X = par_list.original
                        diff_weights_ = par_list[0].get_diff_weights(X)
                        for idx in range(2):
                            diff_weights[idx] = torch.cat([diff_weights[idx], diff_weights_[idx].flatten().cpu()])

                cutoffs = [_get_cutoff(diff_weights[idx], pct) for idx in range(2)]

            for base_module in self.get_encoder_base_modules():
                for n, par_list in list(base_module.parametrizations.items()):
                    X = par_list.original
                    diff_weights = par_list[0].get_diff_weights(X)
                    if pct is not None:
                        diff_masks = [(torch.abs(diff_weights[idx]) > cutoffs[idx]) for idx in range(2)]
                    else:
                        diff_masks = [~torch.isclose(diff_weight, torch.tensor(0.), rtol=1e-8) for diff_weight in diff_weights]
                    parametrize.remove_parametrizations(base_module, n, leave_parametrized=False)
                    for diff_weight, diff_mask in zip(diff_weights, diff_masks):
                        parametrize.register_parametrization(base_module, n, DiffWeightFixmask(diff_weight, diff_mask))

        self.model_state = ModelState.FIXMASK
        self.fixmask_pct = pct


    def _get_sparsity_pen(self, log_ratio: float) -> torch.Tensor:
        assert self.model_state == ModelState.FINETUNING, "model needs to be in finetuning state"
        l0_pen = 0.
        for module_name, base_module in self.get_encoder_base_modules(return_names=True):
            layer_idx = self.get_layer_idx_from_module(module_name)
            sparsity_pen = self.sparsity_pen[layer_idx]
            module_pen = 0.
            for par_list in list(base_module.parametrizations.values()):
                alpha_weights = par_list[0].diff_weight_1.alpha_weights
                if self._debiased:
                    alpha_weights.extend(par_list[0].diff_weight_2.alpha_weights)
                for a in alpha_weights:
                    module_pen += self.get_l0_norm_term(a, log_ratio)
            l0_pen += (module_pen * sparsity_pen)
        return l0_pen