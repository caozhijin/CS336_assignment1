import torch.nn as nn
import torch
from einops import rearrange, einsum

class Rmsnorm(nn.Module):
    def __init__(self, d_model: int,
                  eps: float = 1e-5, 
                  device: torch.device | None = None, 
                  dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model) 
        # and return a tensor of the same shape
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        x = x / rms 
        y = einsum(x, self.weight, '... d_model, d_model -> ... d_model')
        return y.to(in_dtype)