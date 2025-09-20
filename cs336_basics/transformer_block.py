import torch.nn as nn
import torch
from cs336_basics.multihead_self_attention import Multihead_self_attention_with_rope
from cs336_basics.swiglu import SwiGLUFFN
from cs336_basics.rms_norm import Rmsnorm

class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 d_ff: int, max_seq_len: int,
                 theta: float):
        # d_model: int Dimensionality of the Transformer block inputs.
        # num_heads: int Number of heads to use in multi-head self-attention.
        # d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        super().__init__()
        
        self.msa = Multihead_self_attention_with_rope(d_model, num_heads,
                                                      max_seq_len, theta)
        self.ffn = SwiGLUFFN(d_model, d_ff)
        self.norm1 = Rmsnorm(d_model)
        self.norm2 = Rmsnorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: torch.Tensor Input tensor of shape (batch_size, seq_len, d_model)
        # Returns a tensor of shape (batch_size, seq_len, d_model)
        in_type = x.dtype
        x = x.to(torch.float32)
        token_positions = torch.arange(x.size(1)).unsqueeze(0).expand(x.size(0), -1)

        attn_out = self.msa.forward(self.norm1.forward(x), token_positions)
        x = x + attn_out
        ffn_out = self.ffn.forward(self.norm2.forward(x))
        x = x + ffn_out

        x = x.to(in_type)
        return x