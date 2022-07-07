import os
import math
from enum import Enum, auto
from tqdm import tqdm
import contextlib
import torch
from torch.utils.data import DataLoader
import torch.nn.utils.parametrize as parametrize
from transformers import AutoModel

from typing import Union, List, Tuple, Optional, Dict, Callable

from src.models.weight_parametrizations import DiffWeightFinetune, DiffWeightFixmask
from src.utils import dict_to_device

class ModelState(Enum):
    INIT = auto()
    FINETUNING = auto()
    FIXMASK = auto()


class BaseModel(torch.nn.Module):

    @property
    def encoder_module(self) -> torch.nn.Module:
        if isinstance(self.encoder, torch.nn.DataParallel):
            return self.encoder.module
        else:
            return self.encoder

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @property
    def model_type(self) -> str:
        return self.encoder_module.config.model_type

    @property
    def model_name(self) -> str:
        return self.encoder_module.config._name_or_path

    @property
    def hidden_size(self) -> int:
        return self.encoder_module.embeddings.word_embeddings.embedding_dim

    @property
    def total_layers(self) -> int:
        possible_keys = ["num_hidden_layers", "n_layer"]
        cfg = self.encoder_module.config
        for k in possible_keys:
            if k in cfg.__dict__:
                return getattr(cfg, k) + 1 # +1 for embedding layer and last layer
        raise Exception("number of layers of pre trained model could not be determined")

    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, **kwargs)

    def _forward(self, **x) -> torch.Tensor:
        return self.encoder(**x)[0][:,0]

    @torch.no_grad()
    def _evaluate(
        self,
        val_loader: DataLoader,
        forward_fn: Callable,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        label_idx: int = 1,
        desc: str = "",
        **kwargs
        ) -> dict:

        self.eval()

        eval_loss = 0.
        output_list = []
        val_iterator = tqdm(val_loader, desc=f"evaluating {desc}", leave=False, position=1)
        for i, batch in enumerate(val_iterator):

            inputs, labels = batch[0], batch[label_idx]
            if isinstance(inputs, dict):
                inputs = dict_to_device(inputs, self.device)
            else:
                inputs = inputs.to(self.device)
            logits = forward_fn(inputs, **kwargs)
            if isinstance(logits, list):
                eval_loss += torch.stack([loss_fn(x.cpu(), labels) for x in logits]).mean().item()
                preds, _ = torch.mode(torch.stack([pred_fn(x.cpu()) for x in logits]), dim=0)
            else:
                eval_loss += loss_fn(logits.cpu(), labels).item()
                preds = pred_fn(logits.cpu())
            output_list.append((
                preds,
                labels
            ))

        p, l = list(zip(*output_list))
        predictions = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)

        result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
        result["loss"] = eval_loss / (i+1)

        return result

    def _get_mean_loss(self, outputs: Union[torch.Tensor, List[torch.Tensor]], labels: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        losses = []
        for output in outputs:
            losses.append(loss_fn(output, labels))
        return torch.stack(losses).mean()        

    def to(self, device: Union[list, Union[str, torch.device]], *args, **kwargs) -> None:
        self._remove_parallel()
        if isinstance(device, list):
            super().to(device[0])
            if len(device)>1:
                asssert_fn = lambda x: x=="cuda" if isinstance(x, str) else x.type=="cuda"
                assert all([asssert_fn(d) for d in device]), "if list of devices is given, all must be of type 'cuda'"
                self.encoder = torch.nn.DataParallel(self.encoder, device_ids=device)
        else:
            super().to(device)

    def cpu(self):
        self._remove_parallel()
        super().cpu()

    def cuda(self, *args, **kwargs) -> None:
        self._remove_parallel()
        super().cuda(*args, **kwargs)

    def _remove_parallel(self) -> None:
        if isinstance(self.encoder, torch.nn.DataParallel):
            self.encoder = self.encoder.module

    def forward(self, **x) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def _step(self, *args, **kwargs):
        raise NotImplementedError

    def _init_optimizer_and_schedule(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, output_dir: Union[str, os.PathLike], *args, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, os.PathLike], *args, **kwargs) -> torch.nn.Module:
        raise NotImplementedError


class BasePruningModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_state = ModelState.INIT
        self.fixmask_pct = None


    @property
    def parametrized(self) -> bool:
        return (self.model_state == ModelState.FINETUNING or self.model_state == ModelState.FIXMASK)

    @property
    def fixmask_state(self) -> bool:
        return self.model_state == ModelState.FIXMASK

    @property
    def finetune_state(self) -> bool:
        return self.model_state == ModelState.FINETUNING

    @property
    def n_parametrizations(self) -> int:
        return len(list(self.get_encoder_base_modules()[0].parametrizations.values())[0])


    @staticmethod
    def get_log_ratio(concrete_lower: float, concrete_upper: float) -> int:
        return 0 if (concrete_lower == 0) else math.log(-concrete_lower / concrete_upper)


    @staticmethod
    def get_l0_norm_term(alpha: torch.Tensor, log_ratio: float) -> torch.Tensor:
        return torch.sigmoid(alpha - log_ratio).sum()


    def get_encoder_base_modules(self, return_names: bool = False):
        if self.parametrized:
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
    def _count_non_zero_params(self, *args, **kwargs) -> Tuple[int, int, int]:
        assert self.parametrized, "Function only implemented for diff pruning"

        l = [self._count_non_zero_params_for_module(m, *args, **kwargs) for m in self.get_encoder_base_modules()]
        return [sum(x) for x in list(zip(*l))]


    @torch.no_grad()
    def _count_non_zero_params_per_layer(self, *args, **kwargs) -> Dict[int, Tuple[int, int, int]]:
        assert self.parametrized, "Function only implemented for diff pruning"

        t = torch.zeros((self.total_layers, 3), dtype=int)
        for module_name, base_module in self.get_encoder_base_modules(return_names=True):
            layer_idx = self.get_layer_idx_from_module(module_name)
            counts = self._count_non_zero_params_for_module(base_module, *args, **kwargs)
            t[layer_idx] += torch.tensor(counts)
        return {i:v.tolist() for i,v in enumerate(t)}


    @torch.no_grad()
    def _count_non_zero_params_for_module(self, m: torch.nn.Module, idx: Optional[int] = None, merged: bool = False) -> Tuple[int, int, int]:

        def count_fn(p, binary: bool):
            if binary:
                p = p.bool()
                n_p = p.numel()
                n_p_zero = (~p).sum()
                n_p_one = (n_p - n_p_zero)
            else:
                n_p = p.numel()
                n_p_zero = (p == 0.).sum()
                n_p_one = (p == 1.).sum()
            return torch.tensor([n_p, n_p_zero, n_p_one])

        assert hasattr(m, "parametrizations"), "module has no parametrizations"
        p_counts = torch.zeros((3,), dtype=int)
        with self.deterministic():
            for n, par_list in list(m.parametrizations.items()):
                if merged:
                    if isinstance(par_list[0], DiffWeightFixmask):
                        p = torch.stack([x.mask for x in par_list]).sum(0)
                    else:
                        p = torch.stack([(x.z != 0.) for x in par_list]).sum(0)
                    p_counts += count_fn(p, True)
                else:
                    if idx is not None: par_list = [par_list[idx]]
                    for par in par_list:
                        p = par.mask if isinstance(par, DiffWeightFixmask) else par.z
                        p_counts += count_fn(p, p.dtype==torch.bool)
                               
        return p_counts.tolist()


    def _remove_parametrizations(self) -> None:
        self._freeze_parametrizations(True)
        for module in self.get_encoder_base_modules():
            try:
                for n in list(module.parametrizations):
                    parametrize.remove_parametrizations(module, n)
            except AttributeError:
                pass
        self.model_state = ModelState.INIT    


    def _add_diff_parametrizations(self, n_parametrizations: int = 1, p_requires_grad: bool = False, fixmask_init: bool = False, **kwargs) -> None:
        assert not self.parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
        for base_module in self.get_encoder_base_modules(): 
            for n,p in list(base_module.named_parameters()):
                p.requires_grad = p_requires_grad
                for _ in range(n_parametrizations): # number of diff networks to add
                    if fixmask_init:
                        # in case of fixmask init, can only initalize with dummy values
                        parametrize.register_parametrization(base_module, n, DiffWeightFixmask(
                            torch.zeros_like(p), torch.ones_like(p, dtype=bool)
                        ))
                    else:
                        parametrize.register_parametrization(base_module, n, DiffWeightFinetune(p, **kwargs))
        if fixmask_init:
            self.model_state = ModelState.FIXMASK
        else:
            self.model_state = ModelState.FINETUNING


    def _activate_parametrizations(self, active: bool, idx: int):
        for base_module in self.get_encoder_base_modules():
            try:
                for par_list in base_module.parametrizations.values():
                    try:
                        par_list[idx].active = active
                    except IndexError:
                        pass
            except AttributeError:
                pass


    def _freeze_parametrizations(self, frozen: bool, idx: Optional[int] = None):
        for base_module in self.get_encoder_base_modules():
            try:
                for par_list in base_module.parametrizations.values():
                    if idx is not None:
                        try:
                            par_list[idx].set_frozen(frozen)
                        except IndexError:
                            pass
                    else:
                        for par in par_list:
                            par.set_frozen(frozen)
            except AttributeError:
                pass


    def _freeze_original_parameters(self, frozen: bool):
        for base_module in self.get_encoder_base_modules():
            for par_list in base_module.parametrizations.values():
                par_list.original.requires_grad = not frozen


    @torch.no_grad()
    def _finetune_to_fixmask(self, pct: Optional[float] = None, n_parametrizations: int = 1, merged_cutoff: bool = False, merged_min_pct: float = 0.01) -> None:

        def _get_cutoff(values, pct, abs = True):
            k = math.ceil(len(values) * pct)
            if abs: values = torch.abs(values)
            return torch.topk(values, k, largest=True, sorted=True)[0][-1]

        assert self.model_state == ModelState.FINETUNING, "model needs to be in finetuning state"

        with self.deterministic():

            if pct is not None:

                diff_weights_abs = [torch.tensor([])] * n_parametrizations
                for base_module in self.get_encoder_base_modules():
                    for n, par_list in list(base_module.parametrizations.items()):
                        w = par_list.original.detach()
                        for idx in range(n_parametrizations):
                            diff_weight = par_list[idx].diff_weight(w)
                            diff_weights_abs[idx] = torch.cat([diff_weights_abs[idx], torch.abs(diff_weight.flatten().cpu())])
                            w = diff_weight + w

                if merged_cutoff and (n_parametrizations > 1):
                    min_cutoffs = [_get_cutoff(x, merged_min_pct, abs=False) for x in diff_weights_abs]
                    if merged_min_pct >= pct:
                        print(f"merged_min_pct >= pct, using target sparsity merged_min_pct={merged_min_pct}")
                        cutoffs = min_cutoffs
                    else:
                        remaining = torch.cat([x[x<c] for x,c in zip(diff_weights_abs, min_cutoffs)])
                        remaining_cutoff = _get_cutoff(remaining, pct - merged_min_pct)
                        cutoffs = [min(remaining_cutoff, c) for c in min_cutoffs]          
                else:
                    cutoffs = [_get_cutoff(x, pct, abs=False) for x in diff_weights_abs]

            for base_module in self.get_encoder_base_modules():
                for n, par_list in list(base_module.parametrizations.items()):
                    diff_weights = []
                    w = par_list.original
                    for idx in range(n_parametrizations):
                        diff_weight = par_list[idx].diff_weight(w)
                        if pct is not None:
                            i = 0 if merged_cutoff else idx
                            diff_mask = (torch.abs(diff_weight) >= cutoffs[i])
                        else:
                            diff_mask = ~torch.isclose(diff_weight, torch.tensor(0.), rtol=1e-8)
                        diff_weights.append((diff_weight, diff_mask))
                        w = diff_weight + w
                    parametrize.remove_parametrizations(base_module, n, leave_parametrized=False)
                    for (diff_weight, diff_mask) in diff_weights:
                        parametrize.register_parametrization(base_module, n, DiffWeightFixmask(diff_weight, diff_mask))

        self.model_state = ModelState.FIXMASK
        self.fixmask_pct = pct


    def _get_diff_param_groups(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
        learning_rate_alpha: Optional[float] = None,
        idx: Optional[int] = None,
    ) -> list:

        if idx is None:
            idx_len = 0
            idx = ""
        else:
            idx_len = len(str(idx))

        if self.model_state == ModelState.FIXMASK:
            return [
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-(12+idx_len):] == f"{idx}.diff_weight"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                }
            ]
        else:
            return [
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-(9+idx_len):] == f"{idx}.finetune"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                },
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-(6+idx_len):]==f"{idx}.alpha" or n[-(12+idx_len):]==f"{idx}.alpha_group"],
                    "lr": learning_rate_alpha
                }
            ]


    def _get_sparsity_pen(self, log_ratio: float, idx: int) -> torch.Tensor:
        assert self.finetune_state, "model needs to be in finetuning state"
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


    @contextlib.contextmanager
    def deterministic(self):
        tmp_state = self.training
        if tmp_state: self.eval()
        yield
        if tmp_state: self.train()