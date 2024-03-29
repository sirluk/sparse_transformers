import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils import parametrize, parameters_to_vector
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel

from typing import Union, Callable, List, Dict, Generator, Tuple, Optional
from enum import Enum, auto

from src.models.model_heads import ClfHead, AdvHead
from src.models.model_base import BaseDiffModel, ModelState
from src.models.diff_param import DiffWeightFinetune, DiffWeightFixmask
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class DiffModel(BaseDiffModel):     
        
    def __init__(
        self,
        model_name: str,
        num_labels_task: int,
        num_labels_protected: int,
        adv_count: int,
        **kwargs
    ): 
        super().__init__(model_name, **kwargs)

        self.num_labels_task = num_labels_task
        self.num_labels_protected = num_labels_protected
        self.adv_count = adv_count
           
        # heads
        self.task_head = ClfHead(self.hidden_size, num_labels_task)
        self.adv_head = AdvHead(adv_count, adv_rev_ratio, hid_sizes=self.hidden_size, num_labels=num_labels_protected)

        self._model_state = ModelState.INIT
         
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
        metrics: Dict[str, Callable],
        loss_fn_protected: Callable,
        metrics_protected: Dict[str, Callable],
        num_epochs_warmup: int,
        num_epochs_finetune: int,
        num_epochs_fixmask: int,
        alpha_init: Union[int, float],
        concrete_lower: float,
        concrete_upper: float,
        structured_diff_pruning: bool,
        sparsity_pen: Union[float,list],
        fixmask_pct: float,
        learning_rate: float,
        learning_rate_alpha: float,
        learning_rate_adverserial: float,
        optimizer_warmup_steps: int,
        weight_decay: float,
        adam_epsilon: float,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike]
    ) -> None:
       
        self.global_step = 0
        num_epochs_total = num_epochs_warmup + num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) * (num_epochs_warmup + num_epochs_finetune)
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
            learning_rate_alpha,
            learning_rate_adverserial,
            weight_decay,
            adam_epsilon,
            optimizer_warmup_steps
        )

        train_str = "Epoch {}, model_state: {}{}"
        str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k,v in d.items()])
        
        train_iterator = trange(num_epochs_total, desc=train_str.format(0, self._model_state, ""), leave=False, position=0)
        for epoch in train_iterator:
            warmup = (epoch<num_epochs_warmup)        
                
            # switch to fixed mask training
            if epoch == num_epochs_finetune:
                self._finetune_to_fixmask(fixmask_pct)
                self._init_optimizer_and_schedule(
                    train_steps_fixmask,
                    learning_rate,
                    learning_rate_alpha,
                    learning_rate_adverserial,
                    weight_decay,
                    adam_epsilon,
                    optimizer_warmup_steps
                )
            
            self._step(
                train_loader,
                loss_fn,
                logger,
                log_ratio,
                max_grad_norm,
                loss_fn_protected,
                warmup=warmup
            )
                        
            result = self.evaluate(
                val_loader, 
                loss_fn,
                metrics
            )
            if warmup:
                result_debiased = {k:None for k in result.keys()}
                result_protected = {k:None for k in result.keys()}
            else:
                result_debiased = self.evaluate(
                    val_loader, 
                    loss_fn,
                    metrics,
                    debiased=True            
                )
                result_protected = self.evaluate(
                    val_loader, 
                    loss_fn_protected,
                    metrics_protected,
                    predict_prot=True            
                )

            logger.validation_loss(epoch, result, "task")
            logger.validation_loss(epoch, result_protected, suffix="protected")

            # count non zero
            n_p, n_p_zero, n_p_between = self._count_non_zero_params()            
            logger.non_zero_params(epoch, n_p, n_p_zero, n_p_between)
              
            suffix = ", " + ", ".join([
                str_suffix(result, "_task"),
                str_suffix(result_debiased, "_task_debiased"),
                str_suffix(result_protected, "_protected")
            ])
            train_iterator.set_description(
                train_str.format(epoch, self._model_state, suffix), refresh=True
            )

            if ((num_epochs_fixmask > 0) and (self._model_state==ModelState.FIXMASK)) or ((num_epochs_fixmask == 0) and (epoch >= num_epochs_warmup)):
                if logger.is_best(result):
                    cpt = self.save_checkpoint(
                        Path(output_dir),
                        concrete_lower,
                        concrete_upper,
                        structured_diff_pruning
                    )

        print("Final results after " + train_str.format(epoch, self._model_state, suffix))
        return cpt

                
    @torch.no_grad()   
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
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
            
        if not debiased: self._activate_diff_parametrizations(False)

        output_list = []
        val_iterator = tqdm(val_loader, desc=f"evaluating {desc}", leave=False, position=1)
        for i, batch in enumerate(val_iterator):

            inputs, labels = batch[0], batch[label_idx]
            inputs = dict_to_device(inputs, self.device)
            logits = forward_fn(inputs)
            if isinstance(logits, list):
                labels = labels.repeat(len(logits))
                logits = torch.cat(logits, dim=0)         
            output_list.append((
                logits.cpu(),
                labels
            ))
            
        if not debiased: self._activate_diff_parametrizations(True)
                    
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
        max_grad_norm: float,
        loss_fn_protected: Callable,
        warmup: bool
    ) -> None:
                
        self.train()
        
        epoch_str = "training - step {}, loss_biased: {:7.5f}, loss_debiased: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):
            
            inputs, labels_task, labels_protected = batch
            inputs = dict_to_device(inputs, self.device)
            
            ##################################################
            # START STEP TASK
            self._activate_diff_parametrizations(False)    

            outputs = self(**inputs)
            loss = loss_fn(outputs, labels_task.to(self.device))
            loss_biased = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step() # TODO check if step only at end is better
            self.zero_grad()

            self._activate_diff_parametrizations(True)
            # END STEP TASK
            ##################################################
                
            ##################################################
            # START STEP DEBIAS
            self._freeze_task_weights(True)
            
            hidden = self._forward(**inputs)   
            outputs_task = self.task_head(hidden)
            loss = loss_fn(outputs_task, labels_task.to(self.device))
            
            outputs_protected = self.adv_head.forward_reverse(hidden, lmbda=int(not warmup))
            if isinstance(outputs_protected, torch.Tensor):
                outputs_protected = [outputs_protected]
            losses_protected = []
            for output in outputs_protected:
                losses_protected.append(loss_fn_protected(output, labels_protected.to(self.device)))
            loss_protected = torch.stack(losses_protected).mean()
            loss += loss_protected
            
            if self._model_state == ModelState.FINETUNING:
                l0_pen = 0.
                for module_name, base_module in self.get_encoder_base_modules(return_names=True):
                    layer_idx = self.get_layer_idx_from_module(module_name)
                    sparsity_pen = self.sparsity_pen[layer_idx]
                    module_pen = 0.
                    for n, par_list in list(base_module.parametrizations.items()):
                        for a in par_list[0].alpha_weights:
                            module_pen += self.get_l0_norm_term(a, log_ratio)
                    l0_pen += (module_pen * sparsity_pen)
                loss += l0_pen
                
            loss_debiased = loss.item()          
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            self.zero_grad()
            
            self._freeze_task_weights(False)
                    
            # END STEP DEBIAS
            ##################################################
                
            self.scheduler.step()  # Update learning rate schedule            
            
            logger.step_loss(self.global_step, loss_biased, self.scheduler.get_last_lr()[0])
           
            epoch_iterator.set_description(epoch_str.format(step, loss_biased, loss_debiased), refresh=True)
            
            self.global_step += 1          
                    
            
    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
        concrete_lower: Optional[float] = None,
        concrete_upper: Optional[float] = None,
        structured_diff_pruning: Optional[bool] = None
    ) -> None:
        info_dict = {
            "model_name": self.model_name,
            "num_labels_task": self.num_labels_task,
            "model_state": self._model_state,
            "num_labels_protected": self.num_labels_protected,
            "adv_count": self.adv_count,
            "adv_rev_ratio": self.adv_rev_ratio,
            "encoder_state_dict": self.encoder.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict()
        }
        if self._model_state == ModelState.FINETUNING:
            info_dict = {
                **info_dict,
                "concrete_lower": concrete_lower,
                "concrete_upper": concrete_upper,
                "structured_diff_pruning": structured_diff_pruning
            }            

        filename = f"{self.model_name.split('/')[-1]}-{'fixmask' if self._model_state == ModelState.FIXMASK else 'diff_pruning'}.pt"
        filepath = Path(output_dir) / filename
        torch.save(info_dict, filepath)
        return filepath
    
    @staticmethod
    def load_checkpoint(
        filepath: Union[str, os.PathLike],
        remove_parametrizations: bool = False,
        debiased: bool = True,
        map_location=torch.device('cpu')
    ) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)
    
        diff_model = DiffModel(
            info_dict['model_name'],
            info_dict['num_labels_task'],
            info_dict['num_labels_protected'],
            info_dict['adv_count'],
            info_dict['adv_rev_ratio']
        )
        diff_model.adv_head.load_state_dict(info_dict['adv_head_state_dict'])
            
        if info_dict["model_state"] == ModelState.FINETUNING:
            diff_model._add_diff_parametrizations(
                5, # standard value for alpha init, not important here as it will be overwritten by checkpoint
                info_dict['concrete_lower'],
                info_dict['concrete_upper'],
                info_dict['structured_diff_pruning']
            )
        elif info_dict["model_state"] == ModelState.FIXMASK:
            for base_module in diff_model.get_encoder_base_modules():
                for n,p in list(base_module.named_parameters()):
                    p.requires_grad = False
                    parametrize.register_parametrization(base_module, n, DiffWeightFixmask(
                        torch.clone(p), torch.clone(p.bool())
                    ))          
    
        diff_model._model_state = info_dict["model_state"]
        diff_model.encoder.load_state_dict(info_dict['encoder_state_dict'])
        diff_model.task_head.load_state_dict(info_dict['task_head_state_dict'])
        
        diff_model._activate_diff_parametrizations(debiased)
        
        if remove_parametrizations:
            diff_model._remove_diff_parametrizations()
            
        diff_model.eval()
        
        return diff_model
        
                
    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_alpha: float,
        learning_rate_adverserial: float,
        weight_decay: float = 0.0,
        adam_epsilon: float = 1e-08,
        num_warmup_steps: int = 0,

    ) -> None:
                         
        optimizer_params = [
            {
                    "params": [p.original for m in self.get_encoder_base_modules() for p in m.parametrizations.values()],
                    "lr": learning_rate
            },
            {
                    "params": self.task_head.parameters(),
                    "lr": learning_rate
            },
            {
                    "params": self.adv_head.parameters(),
                    "lr": learning_rate_adverserial
            }
        ]
        if self._model_state == ModelState.FIXMASK:
            optimizer_params.extend([
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-13:] == f"0.diff_weight"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                }
            ])
        else:
            optimizer_params.extend([
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-10:] == f"0.finetune"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                },
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-7:]==f"0.alpha" or n[-13:]==f"0.alpha_group"],
                    "lr": learning_rate_alpha
                }
            ])

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
        
        
    def _freeze_task_weights(self, frozen: bool):
        for base_module in self.get_encoder_base_modules():
            for par in base_module.parametrizations.values():
                par.original.requires_grad = not frozen

                
    def _activate_diff_parametrizations(self, active: bool):
        for base_module in self.get_encoder_base_modules():
             for par_list in base_module.parametrizations.values():
                par_list[0].active = active
               
        
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
