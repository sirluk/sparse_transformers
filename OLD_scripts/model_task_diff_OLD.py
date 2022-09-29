import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.nn.parameter import Parameter
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel
)

from typing import Union, Callable, List, Dict, Tuple, Optional
from enum import Enum, auto

from src.models.model_heads import ClfHead
from src.models.model_base import BaseDiffModel, ModelState
from src.models.diff_param import DiffWeightFinetune, DiffWeightFixmask
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class DiffModelTask(BaseDiffModel):          
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        **kwargs
    ): 
        super().__init__(model_name, **kwargs)

        self.num_labels = num_labels

        # head
        self.task_head = ClfHead(self.hidden_size, num_labels)
        
        self._model_state = ModelState.INIT
            
    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        num_epochs_finetune: int,
        num_epochs_fixmask: int,
        alpha_init: Union[int, float],
        concrete_lower: float,
        concrete_upper: float,
        structured_diff_pruning: bool,
        sparsity_pen: Union[float,list],
        fixmask_pct: float,
        weight_decay: float,
        learning_rate: float,
        learning_rate_alpha: float,
        adam_epsilon: float,
        warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike]
    ) -> None:
        
        self.global_step = 0
        num_epochs_total = num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) * num_epochs_finetune
        train_steps_fixmask = len(train_loader) * num_epochs_fixmask
        
        log_ratio = self.get_log_ratio(concrete_lower, concrete_upper)
        
        self.get_sparsity_pen(sparsity_pen)
        self._add_diff_parametrizations(
            alpha_init,
            concrete_lower,
            concrete_upper,
            structured_diff_pruning  
        )       
        
        self._init_optimizer_and_schedule(
            train_steps_finetune,
            learning_rate,
            weight_decay,
            adam_epsilon,
            warmup_steps,
            learning_rate_alpha,
        )
           
        train_str = "Epoch {}, model_state: {}{}"
        str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k,v in d.items()])
        
        train_iterator = trange(num_epochs_total, desc=train_str.format(0, self._model_state, ""), leave=False, position=0)
        for epoch in train_iterator:
            
            # switch to fixed mask training
            if epoch == num_epochs_finetune:
                self._finetune_to_fixmask(fixmask_pct)
                self._init_optimizer_and_schedule(
                    train_steps_fixmask,
                    learning_rate,
                    weight_decay,
                    adam_epsilon,
                    warmup_steps
                )
            
            self._step(
                train_loader,
                loss_fn,
                logger,
                log_ratio,
                max_grad_norm
            )
                        
            result = self.evaluate(
                val_loader, 
                loss_fn,
                metrics
            )

            logger.validation_loss(epoch, result)
            
            # count non zero
            n_p, n_p_zero, n_p_between = self._count_non_zero_params()            
            logger.non_zero_params(epoch, n_p, n_p_zero, n_p_between)
             
            train_iterator.set_description(
                train_str.format(epoch, self._model_state, str_suffix(result)), refresh=True
            )

            if ((num_epochs_fixmask > 0) and (self._model_state==ModelState.FIXMASK)) or (num_epochs_fixmask == 0):
                if logger.is_best(result):
                    self.save_checkpoint(
                        Path(output_dir),
                        concrete_lower,
                        concrete_upper,
                        structured_diff_pruning
                    )
        
        print("Final results after " + train_str.format(epoch, self._model_state, str_suffix(result)))

                
    @torch.no_grad()   
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
    ) -> dict: 
        self.eval()

        output_list = []
        val_iterator = tqdm(val_loader, desc="evaluating", leave=False, position=1)
        for batch in val_iterator:

            inputs, labels = batch[0], batch[1]
            inputs = dict_to_device(inputs, self.device)
            logits = self(**inputs)
            output_list.append((
                logits.cpu(),
                labels
            ))
                    
        p, l = list(zip(*output_list))
        predictions = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)
        
        eval_loss = loss_fn(predictions, labels).item()
        
        result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
        result["loss"] = eval_loss

        return result


    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        log_ratio: float,
        max_grad_norm: float
    ) -> None:
        self.train()
        
        epoch_str = "training - step {}, loss: {:7.5f}, loss without l0 pen: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):
            
            inputs, labels = batch[0], batch[1]
            inputs = dict_to_device(inputs, self.device)
            outputs = self(**inputs)
            loss = loss_fn(outputs, labels.to(self.device))
            
            loss_no_pen = loss.item()
            
            if self._model_state == ModelState.FINETUNING:
                l0_pen = 0.
                for module_name, base_module in self.get_encoder_base_modules(return_names=True):
                    layer_idx = self.get_layer_idx_from_module(module_name)
                    sparsity_pen = self.sparsity_pen[layer_idx]
                    module_pen = 0.
                    for par_list in list(base_module.parametrizations.values()):
                        for a in par_list[0].alpha_weights:
                            module_pen += self.get_l0_norm_term(a, log_ratio)
                    l0_pen += (module_pen * sparsity_pen)
                loss += l0_pen                   
                
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            self.zero_grad()
                
            logger.step_loss(self.global_step, loss, self.scheduler.get_last_lr()[0])
                           
            epoch_iterator.set_description(epoch_str.format(step, loss.item(), loss_no_pen), refresh=True)
            
            self.global_step += 1
            
            
    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
        concrete_lower: float,
        concrete_upper: float,
        structured_diff_pruning: bool
    ) -> None:
        info_dict = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "model_state": self._model_state,
            "encoder_state_dict": self.encoder.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "concrete_lower": concrete_lower,
            "concrete_upper": concrete_upper,
            "structured_diff_pruning": structured_diff_pruning
        } 
            
        filename = f"checkpoint-{self.model_name.split('/')[-1]}-{'fixmask' if self._model_state == ModelState.FIXMASK else 'diff_pruning'}-task.pt"
        filepath = Path(output_dir) / filename
        torch.save(info_dict, filepath)
        return filepath


    @staticmethod
    def load_checkpoint(filepath: Union[str, os.PathLike], remove_parametrizations: bool = False) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=torch.device('cpu'))
            
        diff_model_task = DiffModelTask(
            info_dict['num_labels'],
            info_dict['model_name']
        )
        
        diff_model._add_diff_parametrizations(
            n_parametrizations = 1,
            p_requires_grad = False,
            fixmask_init = (info_dict["model_state"] == ModelState.FIXMASK),
            alpha_init = 5, # standard value for alpha init, not important here as it will be overwritten by checkpoint
            concrete_lower = info_dict['concrete_lower'],
            concrete_upper = info_dict['concrete_upper'],
            structured_diff_pruning = info_dict['structured_diff_pruning']
        )       
    
        diff_model_task.encoder.load_state_dict(info_dict['encoder_state_dict'])
        diff_model_task.classifier.load_state_dict(info_dict['classifier_state_dict'])
        
        if remove_parametrizations:
            diff_model_task._remove_parametrizations()           
        
        diff_model_task.eval()
        
        return diff_model_task
        
                
    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        weight_decay: float,
        adam_epsilon: float,
        num_warmup_steps: int = 0,
        learning_rate_alpha: Optional[float] = None
    ) -> None:

        diff_param_groups = self._get_diff_param_groups(learning_rate, learning_rate_alpha, weight_decay)
        optimizer_param_groups.append({"params": self.classifier.parameters(), "lr": learning_rate})
        
        self.optimizer = AdamW(optimizer_params, eps=adam_epsilon)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
                
    def _add_diff_parametrizations(self, *args) -> None:
        assert not self._parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
        for base_module in self.get_encoder_base_modules():
            for n,p in list(base_module.named_parameters()):
                parametrize.register_parametrization(base_module, n, DiffWeightFinetune(p, *args))   
        self._model_state = ModelState.FINETUNING
          

    @torch.no_grad()
    def _finetune_to_fixmask(self, pct: float) -> None:
        
        def _get_cutoff(values, pct):
            k = int(len(values) * pct)
            return torch.topk(torch.abs(values), k, largest=True, sorted=True)[0][-1]
        
        assert self._model_state == ModelState.FINETUNING, "model needs to be in finetuning state"
        
        self.eval()
        
        diff_weights = torch.tensor([])
        for base_module in self.get_encoder_base_modules():
            for n, par_list in list(base_module.parametrizations.items()):
                diff_weight = (getattr(base_module, n) - par_list.original).detach().cpu()
                diff_weights = torch.cat([diff_weights, diff_weight.flatten()])
        cutoff = _get_cutoff(diff_weights, pct)
        for base_module in self.get_encoder_base_modules():
            for n, par_list in list(base_module.parametrizations.items()):
                task_weight = torch.clone(par_list.original)
                parametrize.remove_parametrizations(base_module, n)
                p = base_module._parameters[n].detach()
                diff_weight = (p - task_weight)
                diff_mask = (torch.abs(diff_weight) >= cutoff)
                base_module._parameters[n] = task_weight
                parametrize.register_parametrization(base_module, n, DiffWeightFixmask(diff_weight, diff_mask))      
                
        self.train()
        self._model_state = ModelState.FIXMASK
