import os
import math
from tqdm import trange, tqdm
from pathlib import Path
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils import parametrize, parameters_to_vector
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel
)

from typing import Union, Callable, List, Dict, Generator, Tuple, Optional
from enum import Enum, auto

from src.training_logger import TrainLogger
from src.masked_param import MaskedWeight, MaskedWeightFixed
from src.utils import dict_to_device
from src.model_heads import ClfHead, AdvHead


class BaseMaskPruningModel(BaseModel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._model_state = ModelState.INIT
     
    
    @property
    def _parametrized(self) -> bool:
        return (self._model_state == ModelState.FINETUNING or self._model_state == ModelState.FIXMASK)
      
        
    @staticmethod
    def get_log_ratio(concrete_lower: float, concrete_upper: float) -> int:
        return 0 if (concrete_lower == 0) else math.log(-concrete_lower / concrete_upper)
    
    
    @staticmethod
    def get_l0_norm_term(alpha: torch.Tensor, log_ratio: float) -> torch.Tensor:
        return torch.sigmoid(alpha - log_ratio).sum()
    
    
    def get_encoder_base_modules(self, return_names: bool = False):
        if self._parametrized:
            check_fn = lambda m: hasattr(m, "parametrizations")
        else:
            check_fn = lambda m: len(m._parameters)>0
        return [(n,m) if return_names else m for n,m in self.encoder.named_modules() if check_fn(m)]   
    

    def get_layer_idx_from_module(self, n: str) -> int:
        # get layer index based on module name
        if self.model_type == "xlnet":
            search_str_emb = "word_embedding"
            search_str_hidden = "layer"
        else:
            search_str_emb = "embeddings"
            search_str_hidden = "encoder.layer"

        if search_str_emb in n:
            return 0
        elif search_str_hidden in n:
            return int(n.split(search_str_hidden + ".")[1].split(".")[0]) + 1
        else:
            return self.total_layers - 1
    
    
    def _init_sparsity_pen(self, sparsity_pen: Union[float, List[float]]) -> None:        
        if isinstance(sparsity_pen, list):
            self.sparsity_pen = sparsity_pen
            assert len(sparsity_pen) == self.total_layers,  "invalid sparsity penalty per layer: # of layers mismatch"
        else:
            self.sparsity_pen = [sparsity_pen] * self.total_layers
    
    
    @torch.no_grad() 
    def _count_non_zero_params(self, idx: int = 0) -> Tuple[int, int]:
        assert self._parametrized, "Function only implemented for diff pruning"
        self.eval()
        n_p = 0
        n_p_zero = 0
        n_p_one = 0
        for base_module in self.get_encoder_base_modules():
            for n, par_list in list(base_module.parametrizations.items()):
                if isinstance(par_list[idx], (DiffWeightFixmask, MaskedWeightFixed)):
                    n_p_ = par_list[idx].mask.numel()
                    n_p_zero_ = (~par_list[idx].mask).sum()
                    n_p += n_p_
                    n_p_zero += n_p_zero_
                    n_p_one += (n_p_ - n_p_zero_)
                else:
                    z = par_list[idx].z.detach()
                    n_p += z.numel()
                    n_p_zero += (z == 0.).sum()
                    n_p_one += (z == 1.).sum()
        self.train()
        
        n_p_between = n_p - (n_p_zero + n_p_one)
        return n_p, n_p_zero, n_p_between
   

    def _remove_parametrizations(self) -> None:
        for module in self.get_encoder_base_modules():
            for n in list(module.parametrizations):
                parametrize.remove_parametrizations(module, n)
        self._model_state = ModelState.INIT
        
        
    def _add_diff_parametrizations(self, n_parametrizations: int = 1, p_requires_grad: bool = False, fixmask_init: bool = False, **kwargs) -> None:
        assert not self._parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
        for base_module in self.get_encoder_base_modules():
            for n,p in list(base_module.named_parameters()):
                p.requires_grad = p_requires_grad
                for _ in range(n_parametrizations): # number of diff networks to add
                    if fixmask_init:
                        # in case of fixmask init, can only initalize with dummy values
                        parametrize.register_parametrization(base_module, n, MaskedWeightFixed(
                            torch.zeros_like(p), torch.ones_like(p, dtype=bool)
                        ))
                    else:
                        parametrize.register_parametrization(base_module, n, DiffWeightFinetune(p, **kwargs))
        if fixmask_init:
            self._model_state = ModelState.FIXMASK
        else:
            self._model_state = ModelState.FINETUNING
        
        
    def _add_mask_parametrizations(self, fixmask_init: bool = False, **kwargs) -> None:
        assert not self._parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
        for base_module in self.get_encoder_base_modules():
            for n,p in list(base_module.named_parameters()):
                if fixmask_init:
                    # in case of fixmask init, can only initalize with dummy values
                    parametrize.register_parametrization(base_module, n, MaskedWeightFixed(
                        torch.ones_like(p, dtype=bool)
                    ))
                else:
                    parametrize.register_parametrization(base_module, n, MaskedWeightFinetune(p, **kwargs))
        if fixmask_init:
            self._model_state = ModelState.FIXMASK
        else:
            self._model_state = ModelState.FINETUNING
        
    def _activate_parametrizations(self, active: bool, idx: int):
        for base_module in self.get_encoder_base_modules():
            for par_list in base_module.parametrizations.values():
                par_list[idx].active = active
                
                
    def _freeze_parametrizations(self, frozen: bool, idx: int):
        for base_module in self.get_encoder_base_modules():
            for par_list in base_module.parametrizations.values():
                par_list[idx].set_frozen(frozen)
        
        
    @torch.no_grad()
    def _finetune_to_fixmask(self, pct: float, n_parametrizations: int = 1) -> None:
        
        def _get_cutoff(values, pct):
            k = int(len(values) * pct)
            return torch.topk(torch.abs(values), k, largest=True, sorted=True)[0][-1]    
        
        assert self._model_state == ModelState.FINETUNING, "model needs to be in finetuning state"
        
        self.eval()

        diff_weights = [torch.tensor([])] * n_parametrizations
        for base_module in self.get_encoder_base_modules():
            for n, par_list in list(base_module.parametrizations.items()):
                w = par_list.original.detach()
                for idx in range(n_parametrizations):
                    w_next = par_list[idx](w).detach()
                    diff_weight = (w_next - w).flatten().cpu()
                    diff_weights[idx] = torch.cat([diff_weights[idx], diff_weight])
                    w = w_next
        
        cutoffs = []
        for idx in range(n_parametrizations):
            cutoffs.append(_get_cutoff(diff_weights[idx], pct))
            
        for base_module in self.get_encoder_base_modules():
            for n, par_list in list(base_module.parametrizations.items()):
                diff_weights = []
                w = par_list.original.detach()
                for idx in range(n_parametrizations):
                    w_next = par_list[idx](w).detach()
                    diff_weight = w_next - w
                    diff_mask = (torch.abs(diff_weight) >= cutoffs[idx])
                    diff_weights.append((diff_weight, diff_mask))
                    w = w_next
                parametrize.remove_parametrizations(base_module, n, leave_parametrized=False)    
                for (diff_weight, diff_mask) in diff_weights:
                    # TODO check if requires_grad needs to be set
                    parametrize.register_parametrization(base_module, n, DiffWeightFixmask(diff_weight, diff_mask))  
                    
        self.train()                
        self._model_state = ModelState.FIXMASK
   

    def _get_diff_param_groups(
        self,
        idx: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        learning_rate_alpha: Optional[float] = None
    ) -> list:       

        if self._model_state == ModelState.FIXMASK:
            return [
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-13:] == f"{idx}.diff_weight"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                }
            ]
        else:
            return [
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-10:] == f"{idx}.finetune"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                },
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-7:]==f"{idx}.alpha" or n[-13:]==f"{idx}.alpha_group"],
                    "lr": learning_rate_alpha
                }
            ]
        
        
    def _get_sparsity_pen(self, idx: int) -> torch.Tensor:
        l0_pen = 0.
        for module_name, base_module in self.get_encoder_base_modules(return_names=True):
            layer_idx = self.get_layer_idx_from_module(module_name)
            sparsity_pen = self.sparsity_pen[layer_idx]
            module_pen = 0.
            for n, par_list in list(base_module.parametrizations.items()):
                for a in par_list[idx].alpha_weights:
                    module_pen += self.get_l0_norm_term(a, log_ratio)
            l0_pen += (module_pen * sparsity_pen)
        return l0_pen        



class TaskMaskingModel(BasePruningModel):
        
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
            weight_decay,
            adam_epsilon,
            warmup_steps,
            learning_rate_alpha,
        )
           
        train_str = "Epoch {}, model_state: {}, {}"
        str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k,v in d.items()])

        self.global_step = 0
        train_iterator = trange(num_epochs, desc=train_str.format(0, self._model_state, str_suffix(None, result_protected)), leave=False, position=0)
        for epoch in train_iterator:
                                    
            self._step(
                train_loader,
                loss_fn,
                loss_fn_protected,
                logger,
                log_ratio,
                max_grad_norm
            )
                        
            result = self.evaluate(
                val_loader, 
                loss_fn,
                metrics
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
                
            train_iterator.set_description(
                train_str.format(epoch, self._model_state, str_suffix(result, result_protected)), refresh=True
            )

        # self._z_to_binary()
        # result = self._fixmask_tuning(
        #     train_loader,
        #     val_loader,
        #     loss_fn,
        #     metrics,
        #     fixmask_tuning_steps,
        #     learning_rate
        # )
        # logger.validation_loss(epoch+1, result, "task")
            
        self.save_checkpoint(
            Path(output_dir),
            concrete_lower,
            concrete_upper,
            structured_diff_pruning
        )
                    
        print("Final results after " + train_str.format(epoch, self._model_state, str_suffix(result, result_protected)))

                
    @torch.no_grad()   
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
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
            
            
    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        loss_fn_protected: Callable,
        logger: TrainLogger,
        log_ratio: float,
        max_grad_norm: float
    ) -> None:
                
        self.train()
        
        epoch_str = "training - step {}, loss: {:7.5f}, loss_debiased: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):
            
            inputs, labels_task, labels_protected = batch
            inputs = dict_to_device(inputs, self.device)
            
            ## step task
            
            self._freeze_masks(True)
            outputs_task = self(**inputs)
            loss_task = loss_fn(outputs_task, labels_task.to(self.device))            
            
            loss_task.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            self.zero_grad()
            
            self._freeze_masks(False)
            
            self._freeze_task_weights(True)
            outputs_task = self(**inputs)
            loss_debias = loss_fn(outputs_task, labels_task.to(self.device))
            
            outputs_protected = self.forward_protected_reverse(**inputs)
            if isinstance(outputs_protected, torch.Tensor):
                outputs_protected = [outputs_protected]
            for output in outputs_protected:
                loss_debias += (loss_fn_protected(output, labels_protected.to(self.device)) / len(outputs_protected))
                
            l0_pen = 0.
            for module_name, base_module in self.get_encoder_base_modules(return_names=True):
                layer_idx = self.get_layer_idx_from_module(module_name)
                sparsity_pen = self.sparsity_pen[layer_idx]
                module_pen = 0.
                for n, par_list in list(base_module.parametrizations.items()):
                    for a in par_list[0].alpha_weights:
                        module_pen += self.get_l0_norm_term(a, log_ratio)
                l0_pen += (module_pen * sparsity_pen)
            loss_debias += l0_pen
            
            loss_debias.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            self.zero_grad()
            
            self._freeze_task_weights(False)
            
            self.scheduler.step()
            
            loss_dict = {
                "task_loss": loss_task.item(),
                "debias_loss": loss_debias.item()
            }
            logger.step_loss(self.global_step, loss_dict)
           
            epoch_iterator.set_description(epoch_str.format(step, loss_task.item(), loss_debias.item()), refresh=True)
            
            self.global_step += 1
            
            
    def _adv_warmup(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        num_steps: int,
        learning_rate: float
    ) -> dict:        
        
        if num_steps == 0: return
        
        self.train()
        
        optimizer = AdamW(self.adv_head.parameters(), lr=learning_rate)
        
        warmup_str = "adverserial warmup - step {} loss: {:7.5f}"
        warmup_iterator = trange(num_steps, desc=warmup_str.format(0, math.nan), position=0)        
        warmup_step = 0
        while warmup_step < num_steps:
            for batch in train_loader:

                inputs, labels_task, labels_protected = batch
                inputs = dict_to_device(inputs, self.device)
                
                loss = 0.
                with torch.no_grad():
                    hidden = self._forward(**inputs)
                outputs = self.adv_head(hidden) 
                if isinstance(outputs, torch.Tensor):
                    outputs = [outputs]
                for output in outputs:
                    loss += (loss_fn(output, labels_protected.to(self.device)) / len(outputs))         

                loss.backward()
                optimizer.step()      
                self.zero_grad() 

                warmup_iterator.set_description(warmup_str.format(warmup_step, loss.item()), refresh=True)
                warmup_iterator.update()

                warmup_step += 1
                
                if warmup_step >= num_steps:
                    break
                    
        warmup_iterator.close()
                    
        return self.evaluate(
            val_loader, 
            loss_fn,
            metrics,
            predict_prot=True            
        )    
    
    def _fixmask_tuning(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        num_steps: int,
        learning_rate: float
    ) -> dict:        
        
        if num_steps == 0: return
        
        self.train()
        
        optimizer = AdamW(list(self.encoder.parameters()) + list(self.task_head.parameters()), lr=learning_rate)
        
        warmup_str = "fixmask tuning - step {} loss: {:7.5f}"
        warmup_iterator = trange(num_steps, desc=warmup_str.format(0, math.nan), position=0)        
        warmup_step = 0
        while warmup_step < num_steps:
            for batch in train_loader:

                inputs, labels_task, labels_protected = batch
                inputs = dict_to_device(inputs, self.device)
                
                outputs = self(**inputs) 
                loss = loss_fn(outputs, labels_protected.to(self.device))       

                loss.backward()
                optimizer.step()      
                self.zero_grad() 

                warmup_iterator.set_description(warmup_str.format(warmup_step, loss.item()), refresh=True)
                warmup_iterator.update()

                warmup_step += 1
                
                if warmup_step >= num_steps:
                    break
                    
        warmup_iterator.close()
                    
        return self.evaluate(
            val_loader, 
            loss_fn,
            metrics
        )
                                    
                    
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
            "num_labels_protected": self.num_labels_protected,
            "adv_count": self.adv_count,
            "adv_rev_ratio": self.adv_rev_ratio,
            "model_state": self._model_state,
            "encoder_state_dict": self.encoder.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict()
        }
        if self._model_state == ModelState.WEIGHT_PRUNING:
            info_dict = {
                **info_dict,
                "concrete_lower": concrete_lower,
                "concrete_upper": concrete_upper,
                "structured_diff_pruning": structured_diff_pruning
            }            

        _filename = f"checkpoint-adv_rev_ratio_{self.adv_rev_ratio:2.1f}.pt"
        torch.save(info_dict, Path(output_dir) / _filename)
        
    
    @staticmethod
    def load_checkpoint(filepath: Union[str, os.PathLike], remove_parametrizations: bool = False) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=torch.device('cpu'))
             
        diff_network = DiffNetwork(
            info_dict['num_labels_task'],
            info_dict['num_labels_protected'],
            info_dict['adv_count'],
            info_dict['adv_rev_ratio'],
            pretrained_model_name_or_path=info_dict['model_name']
        )
           
        if info_dict["model_state"] == ModelState.WEIGHT_PRUNING:
            diff_network._add_parametrizations(
                0,
                info_dict['concrete_lower'],
                info_dict['concrete_upper'],
                info_dict['structured_diff_pruning']
            )
        elif info_dict["model_state"] == ModelState.FIXMASK:
            for base_module in self.get_encoder_base_modules():
                for n,p in list(base_module.named_parameters()):
                    parametrize.register_parametrization(base_module, n, MaskedWeightFixed(torch.clone(p).bool()))            

        diff_network._model_state = info_dict["model_state"]
        diff_network.encoder.load_state_dict(info_dict['encoder_state_dict'])
        diff_network.task_head.load_state_dict(info_dict['task_head_state_dict'])
        diff_network.adv_head.load_state_dict(info_dict['adv_head_state_dict'])
        
        if remove_parametrizations:
            diff_network._remove_parametrizations()
        
        diff_network.eval()
        
        return diff_network
        
                
    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_alpha: float,
        learning_rate_adverserial: float,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-08,
        num_warmup_steps: int = 0
    ) -> None:
        
        is_alpha = lambda param_name: param_name[-7:]==f"0.alpha" or param_name[-13:]==f"0.alpha_group"
                         
        optimizer_params = [
            {
                    "params": self.task_head.parameters(),
                    "lr": learning_rate
            },
            {
                    "params": self.adv_head.parameters(),
                    "lr": learning_rate_adverserial
            },
            {
                "params": [p for n,p in self.encoder.named_parameters() if not is_alpha(n)],
                "weight_decay": weight_decay,
                "lr": learning_rate
            },
            {
                "params": [p for n,p in self.encoder.named_parameters() if is_alpha(n)],
                "lr": learning_rate_alpha
            }
        ]    
            
        self.optimizer = AdamW(optimizer_params, eps=adam_epsilon)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        
        
    def _init_sparsity_pen(self, sparsity_pen: Union[float, List[float]]) -> None:        
        if isinstance(sparsity_pen, list):
            self.sparsity_pen = sparsity_pen
            assert len(sparsity_pen) == self.total_layers,  "invalid sparsity penalty per layer: # of layers mismatch"
        else:
            self.sparsity_pen = [sparsity_pen] * self.total_layers
            
    def _add_parametrizations(self, *args) -> None:
        assert not self._model_state == ModelState.WEIGHT_PRUNING, "cannot add mask parametrizations because of existing parametrizations in the model"
        for base_module in self.get_encoder_base_modules():
            for n,p in list(base_module.named_parameters()):
                parametrize.register_parametrization(base_module, n, MaskedWeight(p, *args))
        
        self._model_state = ModelState.WEIGHT_PRUNING
     
    @weight_pruning_fn
    def _z_to_binary(self):
        self._freeze_masks(True)
        for base_module in self.get_encoder_base_modules():
            for n, par_list in list(base_module.parametrizations.items()):
                z = (par_list[0].z > .5)
                parametrize.remove_parametrizations(base_module, n, leave_parametrized=False)
                parametrize.register_parametrization(base_module, n, MaskedWeightFixed(z))
        self._model_state = ModelState.FIXMASK
        
    @weight_pruning_fn
    def _freeze_task_weights(self, frozen: bool):
        for base_module in self.get_encoder_base_modules():
            for par_list in base_module.parametrizations.values():
                par_list.original.requires_grad = not frozen
                    
    @weight_pruning_fn
    def _freeze_masks(self, frozen: bool):
        for base_module in self.get_encoder_base_modules():
             for par_list in base_module.parametrizations.values():
                par_list[0].frozen = frozen
                                                      
    @torch.no_grad()
    def _count_non_zero_params(self) -> Tuple[int, int]:
        n_p = 0
        n_p_zero = 0
        for base_module in self.get_encoder_base_modules():
            for n, par_list in list(base_module.parametrizations.items()):
                if isinstance(par_list[0], MaskedWeight):
                    par_list[0].frozen = True
                z = par_list[0].z.detach()
                n_p += z.numel()
                n_p_zero += (z == 0.).sum()
                if isinstance(par_list[0], MaskedWeight):
                    par_list[0].frozen = False
        return n_p, n_p_zero

    def _remove_parametrizations(self) -> None:
        for module in self.get_encoder_base_modules():
            for n in list(module.parametrizations):
                parametrize.remove_parametrizations(module, n)
        self._model_state = ModelState.FINETUNING