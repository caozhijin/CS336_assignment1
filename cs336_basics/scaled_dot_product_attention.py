import torch.nn as nn
import torch
from einops import rearrange, einsum
from cs336_basics.softmax import Softmax

class Scaled_dot_product_attention(nn.Module):
    def __init__(self):
        # d_k: int dimension of query and key vectors
        # device: torch.device | None = None Device to store the buffer on
        super().__init__()
        self.softmax = Softmax(dim_id=-1)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Q: torch.Tensor Query tensor of shape (batch_size, ..., seq_len, d_k)
        # K: torch.Tensor Key tensor of shape (batch_size, ..., seq_len, d_k)
        # V: torch.Tensor Value tensor of shape (batch_size, ..., seq_len, d_v)
        # attn_mask: torch.Tensor | None = None Optional attention mask of shape (seq_len, seq_len)
        # Returns a tensor of shape (batch_size, ..., d_v)
        in_type = Q.dtype
        Q = Q.to(torch.float32)
        K = K.to(torch.float32)
        V = V.to(torch.float32)

        self.d_k = Q.shape[-1]
        self.scale = 1.0 / (self.d_k ** 0.5)

        attn_scores = einsum(Q, K, '... i d_k, ... j d_k -> ... i j') * self.scale #shape (..., seq_len_q, seq_len_k)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = self.softmax(attn_scores) #shape (..., seq_len_q, seq_len_k)

        out = einsum(attn_weights, V, '... i j, ... j d_v -> ... i d_v') #shape (..., seq_len_q, d_v)

        out = out.to(in_type)
        return out