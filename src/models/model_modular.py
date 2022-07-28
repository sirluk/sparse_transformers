from lzma import MODE_NORMAL
import os
import math
import copy
from tqdm import trange, tqdm
from pathlib import Path
from functools import reduce
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.utils.parametrize as parametrize
from transformers import get_linear_schedule_with_warmup

from typing import Union, Callable, Dict, Optional

from src.models.model_heads import SwitchHead, AdvHead, ClfHead
from src.models.model_base import BaseModel, ModelState
from src.models.weight_parametrizations import DeltaWeight
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class ModularModel(BaseModel):

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
        adv_task_head: bool = True,
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
        self.adv_task_head = adv_task_head
        self.has_bottleneck = bottleneck
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_dropout = bottleneck_dropout

        # bottleneck layer
        if self.has_bottleneck:
            self.bottleneck = SwitchHead(True, ClfHead, self.hidden_size, bottleneck_dim, dropout=bottleneck_dropout)
            self.in_size_heads = bottleneck_dim
        else:
            self.bottleneck = SwitchHead(True, torch.nn.Identity)
            self.in_size_heads = self.hidden_size

        # heads
        self.task_head = SwitchHead(adv_task_head, ClfHead, [self.in_size_heads]*(task_n_hidden+1), num_labels_task, dropout=task_dropout)
        self.adv_head = AdvHead(adv_count, hid_sizes=[self.in_size_heads]*(adv_n_hidden+1), num_labels=num_labels_protected, dropout=adv_dropout)

        self.model_state = ModelState.INIT
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
        adv_lambda: float,
        learning_rate: float,
        learning_rate_bottleneck: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        optimizer_warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike],
        seed: Optional[int] = None
    ) -> None:

        self.global_step = 0
        num_epochs_finetune += num_epochs_warmup
        num_epochs_total = num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) * num_epochs_finetune
        train_steps_fixmask = len(train_loader) * num_epochs_fixmask

        self.encoder_biased = self.encoder
        self.encoder_debiased = copy.deepcopy(self.encoder_biased)
        self.model_state = ModelState.FINETUNING

        self._init_optimizer_and_schedule(
            train_steps_finetune,
            learning_rate,
            learning_rate_task_head,
            learning_rate_adv_head,
            learning_rate_bottleneck,
            optimizer_warmup_steps
        )

        self.zero_grad()

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
                self._finetune_to_fixmask()
                self._init_optimizer_and_schedule(
                    train_steps_fixmask,
                    learning_rate,
                    learning_rate_task_head,
                    learning_rate_adv_head,
                    learning_rate_bottleneck,
                    optimizer_warmup_steps
                )

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

            result_str = ", " + ", ".join([
                str_suffix(result, "_task"),
                str_suffix(result_debiased, "_task_debiased"),
                str_suffix(result_protected, "_protected")
            ])
            train_iterator.set_description(
                train_str.format(epoch, self.model_state, result_str), refresh=True
            )

            if self.model_state == ModelState.FIXMASK or ((num_epochs_fixmask == 0) and (epoch >= num_epochs_warmup)):
                cpt = self.save_checkpoint(Path(output_dir), seed)

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
            self.set_debiased((not debiased))
        else:
            result = self._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics, label_idx, desc)

        return result


    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
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
            loss_biased = loss_fn(outputs, labels_task.to(self.device))

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

            hidden = self._forward(**inputs)
            outputs_task = self.task_head(hidden)
            loss_task_adv = loss_fn(outputs_task, labels_task.to(self.device))
            loss_debiased += loss_task_adv

            outputs_protected = self.adv_head.forward_reverse(hidden, lmbda=adv_lambda)
            loss_protected = self._get_mean_loss(outputs_protected, labels_protected.to(self.device), loss_fn_protected)
            loss_debiased += loss_protected

            loss_debiased.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            self.zero_grad()
            
            # END STEP DEBIAS
            ##################################################

            # self.scheduler.step()

            losses_dict = {
                "task": loss_biased.item(),
                "task_adv": loss_task_adv.item(),
                "protected": loss_protected.item(),
            }
            logger.step_loss(self.global_step, losses_dict)

            epoch_iterator.set_description(epoch_str.format(
                step,
                losses_dict["task"],
                losses_dict["task_adv"] + losses_dict["protected"]
            ), refresh=True)

            self.global_step += 1

    @torch.no_grad()
    def set_debiased(self, debiased: bool) -> None:
        try:
            check = (debiased != self._debiased)
        except AttributeError:
            check = True
        if check:
            if self.model_state == ModelState.FIXMASK:
                self._activate_parametrizations(debiased)
            elif self.model_state == ModelState.FINETUNING:
                if debiased:
                    self.encoder = self.encoder_debiased
                else:
                    self.encoder = self.encoder_biased
            self.bottleneck.switch_head(not debiased)
            if self.adv_task_head:
                self.task_head.switch_head(not debiased)
            else:
                self.task_head.freeze_parameters(first=True, frozen=debiased)
            self._debiased = debiased


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

        if self.model_state == ModelState.FINETUNING:
            optimizer_params.extend([
            {
                "params": self.encoder_biased.parameters(),
                "lr": learning_rate                
            },
            {
                "params": self.encoder_debiased.parameters(),
                "lr": learning_rate                
            }            
        ])
        elif self.model_state == ModelState.FIXMASK:
            optimizer_params.append({
                "params": self.encoder.parameters(),
                "lr": learning_rate
            })

        self.optimizer = AdamW(optimizer_params, betas=(0.9, 0.999), eps=1e-08)

        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        # )


    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
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
            "adv_task_head": self.adv_task_head,
            "bottleneck": self.has_bottleneck,
            "bottleneck_dim": self.bottleneck_dim,
            "bottleneck_dropout": self.bottleneck_dropout,
            "model_state": self.model_state,
            "bottleneck_state_dict": self.bottleneck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict()
        }

        d = {}
        if self.model_state == ModelState.FINETUNING:
            self.encoder = self.encoder_biased
            d["encoder_biased_state_dict"] = self.encoder_module.state_dict()
            self.encoder = self.encoder_debiased
            d["encoder_debiased_state_dict"] = self.encoder_module.state_dict()
            self.set_debiased(self._debiased)
        elif self.model_state == ModelState.FIXMASK:
            d["encoder_state_dict"] = self.encoder_module.state_dict()
        info_dict = {**info_dict, **d}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename_parts = [
            self.model_name.split('/')[-1],
            "modular_baseline",
            "separate_encoders" if (self.model_state == ModelState.FINETUNING) else None,
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

        if info_dict['model_state'] == ModelState.FIXMASK:
            for base_module in list(cls_instance.encoder.modules()):
                if len(base_module._parameters)>0:
                    for n, p in list(base_module.named_parameters()):
                        parametrize.register_parametrization(base_module, n, DeltaWeight(torch.clone(p)))   
            cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        else:
            cls_instance.encoder_biased = cls_instance.encoder
            cls_instance.encoder_debiased = copy.deepcopy(cls_instance.encoder_biased)
            cls_instance.encoder_biased.load_state_dict(info_dict['encoder_biased_state_dict'])
            cls_instance.encoder_debiased.load_state_dict(info_dict['encoder_debiased_state_dict'])
        cls_instance.model_state = info_dict['model_state']  

        cls_instance.bottleneck.load_state_dict(info_dict['bottleneck_state_dict'])
        cls_instance.adv_head.load_state_dict(info_dict['adv_head_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])

        cls_instance.set_debiased(debiased)

        if remove_parametrizations and cls_instance.model_state == ModelState.FIXMASK:
            cls_instance._remove_parametrizations()

        cls_instance.eval()

        return cls_instance


    @torch.no_grad()
    def _finetune_to_fixmask(self) -> None:
        assert self.model_state == ModelState.FINETUNING, "model needs to be in finetuning state"
        for module_name, base_module in list(self.encoder_biased.named_modules()):
            if len(base_module._parameters)>0:
                module_debiased = reduce(lambda x, y: getattr(x, y), [self.encoder_debiased] + module_name.split("."))
                for n, p in list(base_module.named_parameters()):
                    delta_weight = (getattr(module_debiased, n) - p).detach()
                    parametrize.register_parametrization(base_module, n, DeltaWeight(delta_weight))
        self.encoder = self.encoder_biased
        self.encoder_biased = None
        self.encoder_debiased = None
        self.model_state = ModelState.FIXMASK


    def _remove_parametrizations(self) -> None:
        assert self.model_state == ModelState.FIXMASK, "model needs to be in fixmask state"
        for module in list(self.encoder.modules()):
            try:
                for n in list(module.parametrizations):
                    parametrize.remove_parametrizations(module, n)
            except AttributeError:
                pass
        self.model_state = ModelState.INIT


    def _activate_parametrizations(self, active: bool):
        assert self.model_state == ModelState.FIXMASK, "model needs to be in fixmask state"
        for module in list(self.encoder.modules()):
            try:
                for par_list in module.parametrizations.values():
                    par_list[0].active = active
            except AttributeError:
                pass
