from enum import Enum, auto
import contextlib
import torch
from  torch import nn
from torch.nn.parameter import Parameter

from typing import Tuple, Optional

from src.utils import concrete_stretched


class DiffWeightFinetune(nn.Module):

    def __init__(self, weight, alpha_init, concrete_lower, concrete_upper, structured):
        super().__init__()
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        self.structured = structured

        self.register_parameter("finetune", Parameter(torch.clone(weight)))
        self.register_parameter("alpha", Parameter(torch.zeros_like(weight) + alpha_init))

        if structured:
            self.register_parameter("alpha_group", Parameter(torch.zeros((1,), device=weight.device) + alpha_init))

        self.active = True

    def forward(self, X):
        if not self.active: return X
        diff = (self.finetune - X).detach()
        return (self.finetune - diff) + self.diff_weight(X)

    def diff_weight(self, X):
        return self.z * (self.finetune - X)

    @property
    def z(self) -> Parameter:
        z = self.dist(self.alpha)
        if self.structured:
            z *= self.dist(self.alpha_group)
        return z

    @property
    def alpha_weights(self) -> list:
        alpha = [self.alpha]
        if self.structured:
            alpha.append(self.alpha_group)
        return alpha

    def dist(self, alpha) -> torch.Tensor:
        return concrete_stretched(
            alpha,
            l=self.concrete_lower,
            r=self.concrete_upper,
            deterministic=(not self.training)
        )

    def set_frozen(self, frozen: bool) -> None:
        self.finetune.requires_grad = not frozen
        self.alpha.requires_grad = not frozen
        if self.structured:
            self.alpha_group.requires_grad = not frozen
        if frozen:
            self.eval()
        else:
            self.train()


class DiffWeightFixmask(nn.Module):

    def __init__(self, diff_weight: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.register_parameter("diff_weight", Parameter(diff_weight * mask))
        self.register_parameter("mask", Parameter(mask, requires_grad=False))
        self.active = True

    def forward(self, X):
        if not self.active: return X
        return X + self.mask * self.diff_weight

    def set_frozen(self, frozen: bool) -> None:
        self.diff_weight.requires_grad = not frozen


class DoubleDiffWeightFinetune(nn.Module):

    class WeightState(Enum):
        TRAIN_FIRST = auto()
        TRAIN_SECOND = auto()

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.diff_weight_1 = DiffWeightFinetune(*args, **kwargs)
        self.diff_weight_2 = DiffWeightFinetune(*args, **kwargs)
        self.register_parameter("testpar", Parameter(torch.zeros((1,), device=args[0].device)))
        self.set_weight_state()

    def forward(self, X):
        if self.weight_state == self.WeightState.TRAIN_FIRST:
            return self.diff_weight_1(X)
        else:
            with torch.no_grad(), self.deterministic():
                diff_first = self.diff_weight_1.diff_weight(X)
                temp_second = self.diff_weight_2.finetune - X - diff_first
            diff_second = self.diff_weight_2.diff_weight((X+diff_first))
            return (self.diff_weight_2.finetune - temp_second) + self.diff_weight_1.z * diff_second

    @torch.no_grad()
    def get_diff_weights(self, X: Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
        with self.deterministic():
            diff_first = self.diff_weight_1.diff_weight(X)
            diff_second = self.diff_weight_1.diff_weight((X + diff_first))
            return diff_first, self.diff_weight_1.z * diff_second

    @torch.no_grad()
    def get_z_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with self.deterministic():
            return self.diff_weight_1.z, self.diff_weight_1.z * self.diff_weight_2.z

    def set_weight_state(self, first: bool = True) -> None:
        if first:
            self.diff_weight_1.set_frozen(False)
            self.diff_weight_2.set_frozen(True)
            self.weight_state = self.WeightState.TRAIN_FIRST
        else:
            self.diff_weight_1.set_frozen(False)
            self.diff_weight_1.finetune.requires_grad = False
            self.diff_weight_2.set_frozen(False)
            self.weight_state = self.WeightState.TRAIN_SECOND

    @contextlib.contextmanager
    def deterministic(self):
        tmp_state = self.training
        if tmp_state: self.eval()
        yield
        if tmp_state: self.train()


# class DoubleDiffWeightFinetune(nn.Module):

#     class WeightState(Enum):
#         TRAIN_FIRST = auto()
#         TRAIN_SECOND = auto()

#     def __init__(self, weight, alpha_init, concrete_lower, concrete_upper, structured):
#         super().__init__()
#         self.concrete_lower = concrete_lower
#         self.concrete_upper = concrete_upper
#         self.structured = structured

#         for i in range(2):
#             self.register_parameter(f"finetune{i}", Parameter(torch.clone(weight)))
#             self.register_parameter(f"alpha{i}", Parameter(torch.zeros_like(weight) + alpha_init))

#             if structured:
#                 self.register_parameter(f"alpha_group{i}", Parameter(torch.zeros((1,), device=weight.device) + alpha_init))
        
#         self.set_weight_state()

#     def forward(self, X):
#         if self.weight_state == self.WeightState.TRAIN_FIRST:
#             with torch.no_grad():
#                 diff = (self.finetune0 - X)
#             return (self.finetune0 - diff) + self.z(0) * (self.finetune0 - X)
#         else:
#             with torch.no_grad(), self.deterministic():
#                 diff_first = self.z(0) * (self.finetune0 - X)
#                 temp_second = self.finetune1 - X - diff_first        
#             return (self.finetune1 - temp_second) + self.z(0) * self.z(1) * (self.finetune1 - X - diff_first)

#     def z(self, idx: int) -> Parameter:
#         z = self.dist(getattr(self, f"alpha{idx}"))
#         if self.structured:
#             z *= self.dist(getattr(self, f"alpha_group{idx}"))
#         return z    

#     def dist(self, alpha) -> torch.Tensor:
#         return concrete_stretched(
#             alpha,
#             l=self.concrete_lower,
#             r=self.concrete_upper,
#             deterministic=(not self.training)
#         )

#     @torch.no_grad()
#     def get_diff_weights(self, X: Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
#         with self.deterministic():
#             diff_first = self.z(0) * (self.finetune0 - X)
#             diff_second = self.z(0) * self.z(1) * (self.finetune1 - X - diff_first)
#             return diff_first, diff_second

#     @torch.no_grad()
#     def get_z_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         with self.deterministic():
#             return self.z(0), self.z(1)

#     def set_weight_state(self, first: bool = True) -> None:
#         if first:
#             self.weight_state = self.WeightState.TRAIN_FIRST
#         else:
#             self.weight_state = self.WeightState.TRAIN_SECOND

#     @contextlib.contextmanager
#     def deterministic(self):
#         tmp_state = self.training
#         if tmp_state: self.eval()
#         yield
#         if tmp_state: self.train()