import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel

from typing import Union, Callable, Dict, Optional

from src.models.model_heads import ClfHead, AdvHead
from src.models.model_base import BaseModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class ParametrizedWeight(nn.Module):
    
    def __init__(self, weight):
        super().__init__()  
        self.register_parameter("finetune", Parameter(weight))      
        self.active = True
          
    def forward(self, X):
        if self.active: 
            return X + self.finetune
        else:
            return X


class AdvModel(BaseModel):
    
    def __init__(
        self,
        model_name: str,
        num_labels_task: int,
        num_labels_protected: int,
        dropout: float = .3,
        adv_dropout: float = .3,
        adv_count: int = 5,
        **kwargs
    ): 
        super().__init__(model_name, **kwargs)
        
        self.num_labels_task = num_labels_task
        self.num_labels_protected = num_labels_protected
        self.dropout = dropout
        self.adv_dropout = adv_dropout
        self.adv_count = adv_count
        
        # heads
        self.task_head = ClfHead(self.hidden_size, num_labels_task, dropout=dropout)
        self.adv_head = AdvHead(adv_count, hid_sizes=self.hidden_size, num_labels=num_labels_protected, dropout=adv_dropout)
        
        self._bitfit = False

    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))
    
    def forward_protected(self, **x) -> torch.Tensor:
        return self.adv_head(self._forward(**x)) 
    
    def fit(
        self,
        bitfit,
        *args,
        **kwargs
    ) -> None:
        if bitfit:
            self._fit_bitfit(*args, **kwargs)
            self._bitfit = True
        else:
            self._fit_simple(*args, **kwargs)
          
        
    def evaluate(self, *args, **kwargs):
        if self._bitfit:
            self._evaluate_bitfit(*args, **kwargs)
        else:
            self._evaluate_simple(*args, **kwargs)
            
        
    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike]
    ) -> None:
        info_dict = {
            "model_name": self.model_name,
            "num_labels_task": self.num_labels_task,
            "num_labels_protected": self.num_labels_protected,
            "dropout": self.dropout,
            "adv_dropout": self.adv_dropout,
            "adv_count": self.adv_count,
            "bitfit": self._bitfit,
            "encoder_state_dict": self.encoder.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict()
        }          

        filename = f"{self.model_name.split('/')[-1]}-adv_baseline{'-bitfit' if self._bitfit else ''}.pt"
        filepath = Path(output_dir) / filename
        torch.save(info_dict, filepath)
        return filepath
        
        
    @classmethod
    def load_checkpoint(cls, filepath: Union[str, os.PathLike], map_location: Union[str, torch.device] = torch.device('cpu')) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)
    
        cls_instance = cls(
            info_dict['model_name'],
            info_dict['num_labels_task'],
            info_dict['num_labels_protected'],
            info_dict['dropout'],
            info_dict['adv_dropout'],
            info_dict['adv_count']
        )
        
        if info_dict['bitfit']:
            cls_instance._bitfit_init()
        
        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.adv_head.load_state_dict(info_dict['adv_head_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])
        
        cls_instance.eval()
        
        return cls_instance
   

    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_adverserial: float,
        num_warmup_steps: int = 0
    ) -> None:
        optimizer_params = [
            {
                "params": self.encoder.parameters(),
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

        self.optimizer = AdamW(optimizer_params, betas=(0.9, 0.999), eps=1e-08)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    
    def _fit_simple(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        loss_fn_protected: Callable,
        metrics_protected: Dict[str, Callable],
        num_epochs: int,
        num_epochs_warmup: int,
        learning_rate: float,
        learning_rate_adverserial: float,
        warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike]
    ) -> None:
        
        self.global_step = 0
        num_epochs_total = num_epochs + num_epochs_warmup
        train_steps = len(train_loader) * num_epochs_total
        
        self._init_optimizer_and_schedule(
            train_steps,
            learning_rate,
            learning_rate_adverserial,
            warmup_steps
        )
        
        self.zero_grad()
        
        train_str = "Epoch {}, {}"
        str_suffix = lambda d: ", ".join([f"{k}: {v}" for k,v in d.items()])
        
        train_iterator = trange(num_epochs_total, desc=train_str.format(0, ""), leave=False, position=0)
        for epoch in train_iterator:
            warmup = (epoch<num_epochs_warmup)
               
            self._step(
                train_loader,
                loss_fn,
                logger,
                max_grad_norm,
                loss_fn_protected,
                warmup=warmup
            )
                        
            result = self.evaluate(
                val_loader, 
                loss_fn,
                metrics
            )
            logger.validation_loss(epoch, result, "task")
            
            result_protected = self.evaluate(
                val_loader, 
                loss_fn_protected,
                metrics_protected,
                predict_prot=True
            ) 
            logger.validation_loss(epoch, result_protected, suffix="protected")
            
            train_iterator.set_description(
                train_str.format(epoch, str_suffix(result_protected)), refresh=True
            )
            
            if logger.is_best(result):
                cpt = self.save_checkpoint(Path(output_dir))
        
        print("Final results after " + train_str.format(epoch, str_suffix(result)))
        return cpt
        
        
    @torch.no_grad()   
    def _evaluate_simple(
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

        p, l = list(zip(*output_list))
        predictions = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)

        eval_loss = loss_fn(predictions, labels).item()

        result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
        result["loss"] = eval_loss

        return result  
    
            
    def _step_simple(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        max_grad_norm: float,
        loss_fn_protected: Callable,
        warmup: bool
    ) -> None:
        self.train()
        
        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):
            
            inputs, labels_task, labels_protected = batch
            inputs = dict_to_device(inputs, self.device)
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
            loss.backward()   

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()          
            self.zero_grad()    
                
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            logger.step_loss(self.global_step, loss, lr) 
                           
            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)
            
            self.global_step += 1

    
    def _fit_bitfit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        loss_fn_protected: Callable,
        metrics_protected: Dict[str, Callable],
        num_epochs: int,
        num_epochs_warmup: int,
        learning_rate: float,
        learning_rate_adverserial: float,
        warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike]
    ) -> None:
        
        # bitfit mode
        self._bitfit_init()
        
        self.global_step = 0
        num_epochs_total = num_epochs + num_epochs_warmup
        train_steps = len(train_loader) * num_epochs_total
        
        self._init_optimizer_and_schedule(
            train_steps,
            learning_rate,
            learning_rate_adverserial,
            warmup_steps
        )
        
        self.zero_grad()
        
        train_str = "Epoch {}, {}"
        str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k,v in d.items()])
        
        train_iterator = trange(num_epochs_total, desc=train_str.format(0, ""), leave=False, position=0)
        for epoch in train_iterator:
            warmup = (epoch<num_epochs_warmup)
               
            self._step(
                train_loader,
                loss_fn,
                logger,
                max_grad_norm,
                loss_fn_protected,
                warmup=warmup
            )
                        
            result = self.evaluate(
                val_loader, 
                loss_fn,
                metrics
            )
            logger.validation_loss(epoch, result, "task")
            
            result_protected = self.evaluate(
                val_loader, 
                loss_fn_protected,
                metrics_protected,
                predict_prot=True            
            )
            logger.validation_loss(epoch, result_protected, suffix="protected")
            
            if warmup:
                result_debiased = {k:None for k in result.keys()}
            else:
                result_debiased = self.evaluate(
                    val_loader, 
                    loss_fn,
                    metrics,
                    debiased=True            
                )
                logger.validation_loss(epoch, result_protected, suffix="task_debiased")
            
            suffix = ", " + ", ".join([
                str_suffix(result, "_task"),
                str_suffix(result_debiased, "_task_debiased"),
                str_suffix(result_protected, "_protected")
            ])
            train_iterator.set_description(
                train_str.format(epoch, suffix), refresh=True
            )

            if logger.is_best(result):
                cpt = self.save_checkpoint(Path(output_dir))
        
        print("Final results after " + train_str.format(epoch, suffix))
        return cpt
        
        
    @torch.no_grad()   
    def _evaluate_bitfit(
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
            
        if not debiased: self._activate_parametrizations(False)

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
            
        if not debiased: self._activate_parametrizations(True)

        p, l = list(zip(*output_list))
        predictions = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)

        eval_loss = loss_fn(predictions, labels).item()

        result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
        result["loss"] = eval_loss

        return result  
    
            
    def _step_bitfit(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
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
            self._activate_parametrizations(False)

            outputs = self(**inputs)
            loss = loss_fn(outputs, labels_task.to(self.device))
            
            if self._model_state == ModelState.FINETUNING:
                loss += self._get_sparsity_loss(0)
            
            loss_biased = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step() # TODO check if step only at end is better
            self.zero_grad()

            self._activate_parametrizations(True)
            # END STEP TASK
            ##################################################
            
            ##################################################
            # START STEP DEBIAS
            self._freeze_pretrained_weights(True)
            
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
                
            loss_debiased = loss.item()          
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            self.zero_grad()
            
            self._freeze_pretrained_weights(False)
            # END STEP DEBIAS
            ##################################################            

            self.scheduler.step()          

            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            logger.step_loss(self.global_step, loss, lr) 
                           
            epoch_iterator.set_description(epoch_str.format(step, loss_biased, loss_debiased), refresh=True)
            
            self.global_step += 1
    
    
    def _bitfit_init(self):
        base_modules = [m for m in self.encoder.modules() if len(m._parameters)>0]
        for base_module in base_modules:
            module_copy = copy.deepcopy(base_module)
            if hasattr(module_copy, "reset_parameters"):
                module_copy.reset_parameters()
            else:
                raise Exception(f"Module of type {type(module_copy)} has no attribute reset_parameters")
            for n,p in list(module_copy.named_parameters()):
                if "bias" in n:
                    parametrize.register_parametrization(base_module, n, ParametrizedWeight(p))
                else:
                    p.requires_grad = False
        self._bitfit = True
        
        
    def _activate_parametrizations(self, active: bool):
        base_modules = [m for m in self.encoder.modules() if hasattr(m, "parametrizations")]
        for base_module in self.get_encoder_base_modules():
            for par_list in base_module.parametrizations.values():
                par_list[0].active = active
                
                
    def _freeze_pretrained_weights(self, frozen: bool):
        base_modules = [m for m in self.encoder.modules() if hasattr(m, "parametrizations")]
        for base_module in self.get_encoder_base_modules():
            for par_list in base_module.parametrizations.values():
                par_list.original.requires_grad = (not frozen)

