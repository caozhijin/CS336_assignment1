import torch.nn as nn
import torch
from einops import rearrange, einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device | None = None):
        # theta: float Θ value for the RoPE
        # d_k: int dimension of query and key vectors
        # max_seq_len: int Maximum sequence length that will be inputted
        # device: torch.device | None = None Device to store the buffer on
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        self.k_max = d_k // 2
        seq_k = torch.arange(self.k_max,dtype=torch.float32, device=device)
        seq_1 = 1.0 / (theta ** (seq_k / self.k_max))
        seq_2 = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        seq_theta = einsum(seq_2, seq_1, 'i, j -> i j')

        cos = torch.cos(seq_theta)
        sin = torch.sin(seq_theta)

        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        # You should assume that the token positions are a tensor of shape (..., seq_len) 
        # specifying the token positions of x along the sequence dimension.
        in_type = x.dtype
        x = x.to(torch.float32)

        x_pair = rearrange(x, '... seq_len (pair_num l) -> ... seq_len pair_num l', l=2) #shape (..., seq_len, d_k//2, 2)
        cos = self.cos_cached[token_positions] #shape (..., seq_len, d_k//2)
        sin = self.sin_cached[token_positions]
        x1, x2 = x_pair.unbind(-1) #each of shape (..., seq_len, d_k//2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        x_rot = torch.stack((x1_rot, x2_rot), dim=-1) #shape (..., seq_len, d_k//2, 2)
        y = rearrange(x_rot, '... seq_len pair_num l -> ... seq_len (pair_num l)', l=2) #shape (..., seq_len, d_k)
        y = y.to(in_type)
        return y