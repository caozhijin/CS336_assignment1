import torch.nn as nn
import torch
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, 
                out_features: int, 
                device:  torch.device | None = None, 
                dtype: torch.dtype | None = None):
        # in_features: int final dimension of the input
        # out_features: int final dimension of the output
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = (2.0 / (in_features + out_features)) ** 0.5         
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')
        return y
