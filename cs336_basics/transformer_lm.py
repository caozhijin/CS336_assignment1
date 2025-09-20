import torch.nn as nn
import torch
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import Transformer_block
from cs336_basics.rms_norm import Rmsnorm
from cs336_basics.linear import Linear

class Transformer_lm(nn.Module):
    def __init__(self, vocab_size: int,
                       context_length: int,
                       d_model: int,
                       num_layers: int,
                       num_heads: int,
                       d_ff: int,
                       rope_theta: float):
        # vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token
        # embedding matrix.
        # context_length: int The maximum context length, necessary for determining the dimensionality of
        # the position embedding matrix. = max_seq_len
        # num_layers: int The number of Transformer blocks to use.
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            Transformer_block(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.norm = Rmsnorm(d_model, eps=1e-5)
        self.linear = Linear(d_model, vocab_size)

        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: torch.Tensor Input tensor of shape (batch_size, seq_len, d_model)
        # token_positions: torch.Tensor Tensor of shape (batch_size, seq_len) containing the position indices for each token in the input sequences
        # Returns a tensor of shape (batch_size, seq_len, vocab_size)
        x = self.token_embedding.forward(x)
        for i in range(self.num_layers):
            x = self.transformer_blocks[i].forward(x)
        x = self.norm.forward(x)
        logits = self.linear.forward(x)

        return logits