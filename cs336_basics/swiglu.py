import torch
import torch.nn as nn
from .linear import Linear
from einops import rearrange, einsum

def SiLu(x: torch.Tensor) -> torch.Tensor:
    in_type = x.dtype
    x = x.to(torch.float32)
    y = x * torch.sigmoid(x)
    return y.to(in_type)

class SwiGLU(nn.Module):
    def __init__(self,
                 d_model: int, 
                 d_ff: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.linear2_weight = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.linear3_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        std = (2.0 / (d_ff + d_model)) ** 0.5
        nn.init.trunc_normal_(self.linear1_weight, mean=0.0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.linear2_weight, mean=0.0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.linear3_weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        w1x = einsum(self.linear1_weight, x, 'd_ff d_model, d_model -> d_ff')
        w3x = einsum(self.linear3_weight, x, 'd_ff d_model, d_model -> d_ff')
        silu_w1x = SiLu(w1x)
        glu = silu_w1x * w3x
        y = einsum(self.linear2_weight, glu, 'd_model d_ff, d_ff -> d_model')
        return y.to(in_type) 
    


# A variant of SwiGLU that uses Linear layers
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff , d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        gate = SiLu(self.w1.forward(x)) * self.w3.forward(x)
        y = self.w2.forward(gate)
        return y.to(in_type)