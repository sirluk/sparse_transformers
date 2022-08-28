import torch
from  torch import nn
from torch.nn.parameter import Parameter

from src.utils import concrete_stretched          
            
            
class MaskedWeightFinetune(nn.Module):
    
    def __init__(self, weight, alpha_init, concrete_lower, concrete_upper, structured):
        super().__init__()
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        self.structured = structured

        self.register_parameter("alpha", Parameter(torch.zeros_like(weight) + alpha_init))
        
        if structured:
            self.register_parameter("alpha_group", Parameter(torch.zeros((1,), device=weight.device) + alpha_init))
            
        self.active = True
                
    def forward(self, X):
        if self.active:
            return self.z * X
        else:
            return X
    
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
        self.alpha.requires_grad = not frozen
        if self.structured:
            self.alpha_group.requires_grad = not frozen
        if frozen:
            self.eval()
        else:
            self.train()

    
class MaskedWeightFixed(nn.Module):
    
    def __init__(self, mask):
        super().__init__()
        self.register_parameter("mask", Parameter(mask, requires_grad=False))
        
        self.active = True
                
    def forward(self, X):
        if self.active:
            return self.mask * X
        else:
            return X

    def set_frozen(self, frozen: bool) -> None:
        pass