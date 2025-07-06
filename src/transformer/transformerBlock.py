import torch
import torch.nn as nn
from attention import MultiHeadAttention
from layerNorm import LayerNorm
from feedForward import FeedForward

class TransformerBlock(nn.Module):
    """
    Transformer Block that consists of multi-head attention, layer normalization, and feed-forward network.
    This block is typically used in transformer architectures to process sequences of tokens.
    
    Args:
        config (dict): Configuration dictionary containing model parameters.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=config['emb_dim'],
            d_out=config['emb_dim'],
            context_len=config['context_length'],
            dropout=config['drop_rate'],
            num_heads=config['num_heads'],
            qkv_bias=config['qkv_bias']
        )
        self.ln1 = LayerNorm(config['emb_dim'])
        self.ffn = FeedForward(config['emb_dim'], activation=config['activation'])
        self.ln2 = LayerNorm(config['emb_dim'])
        self.drop_shortcut = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x