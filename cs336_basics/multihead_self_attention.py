import torch.nn as nn
import torch
from einops import rearrange, einsum
from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import Scaled_dot_product_attention

class Multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        # d_model: int Dimension of the input and output feature vectors
        # num_heads: int Number of attention heads
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_Q = Linear(d_model, num_heads * self.d_k)
        self.W_K = Linear(d_model, num_heads * self.d_k)
        self.W_V = Linear(d_model, num_heads * self.d_v)
        self.W_O = Linear(num_heads * self.d_v, d_model) 

        self.sdpa = Scaled_dot_product_attention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: torch.Tensor Input tensor of shape (batch_size, seq_len, d_model)
        # Returns a tensor of shape (batch_size, seq_len, d_model)
        
        in_type = x.dtype
        x = x.to(torch.float32)
        batch_size, seq_len, _ = x.size()

        Q = self.W_Q.forward(x) #shape (batch_size, seq_len, num_heads * d_k)
        K = self.W_K.forward(x) #shape (batch_size, seq_len, num_heads * d_k)
        V = self.W_V.forward(x) #shape (batch_size, seq_len, num_heads * d_v)
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)

        attn_mask = torch.ones(batch_size, seq_len, seq_len, device=x.device)
        attn_mask = attn_mask.tril(diagonal=0)#tril 下三角

        heads = self.sdpa.forward(Q, K, V, attn_mask) 
        heads = rearrange(heads, 'b h s v -> b s (h v)')
        out = self.W_O.forward(heads) #shape (batch_size, seq_len, d_model)

        out = out.to(in_type)
        return out
    
class Multihead_self_attention_with_rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 max_seq_len: int, theta: float):
        # d_model: int Dimension of the input and output feature vectors
        # num_heads: int Number of attention heads
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_Q = Linear(d_model, num_heads * self.d_k)
        self.W_K = Linear(d_model, num_heads * self.d_k)
        self.W_V = Linear(d_model, num_heads * self.d_v)
        self.W_O = Linear(num_heads * self.d_v, d_model)

        self.rope = RoPE(theta, self.d_v, max_seq_len)

        self.sdpa = Scaled_dot_product_attention()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: torch.Tensor Input tensor of shape (batch_size, seq_len, d_model)
        # Returns a tensor of shape (batch_size, seq_len, d_model)
        
        in_type = x.dtype
        x = x.to(torch.float32)
        batch_size, seq_len, _ = x.size()

        Q = self.W_Q.forward(x) #shape (batch_size, seq_len, num_heads * d_k)
        K = self.W_K.forward(x) #shape (batch_size, seq_len, num_heads * d_k)
        V = self.W_V.forward(x) #shape (batch_size, seq_len, num_heads * d_v)
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads)
        Q = self.rope.forward(Q, token_positions) 
        K = self.rope.forward(K, token_positions) 
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)

        attn_mask = torch.ones(batch_size, seq_len, seq_len, device=x.device)
        attn_mask = attn_mask.tril(diagonal=0)#tril 下三角

        heads = self.sdpa.forward(Q, K, V, attn_mask) 
        heads = rearrange(heads, 'b h s v -> b s (h v)')
        out = self.W_O.forward(heads) #shape (batch_size, seq_len, d_model)
        
        out = out.to(in_type)
        return out
