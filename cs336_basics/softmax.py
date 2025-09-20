import torch.nn as nn
import torch

class Softmax(nn.Module):
    def __init__(self, dim_id: int):
        # dim: int Dimension along which to compute the softmax
        super().__init__()
        self.dim_id = dim_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        x_max = x.amax(dim=self.dim_id, keepdim=True)
        x_exp = torch.exp(x - x_max)
        out = x_exp / x_exp.sum(dim=self.dim_id, keepdim=True)
        
        out = out.to(in_type)
        return out