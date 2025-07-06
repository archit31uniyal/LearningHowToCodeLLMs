import tiktoken
import torch
from utils.config import *
import torch.nn as nn
from transformerBlock import TransformerBlock
from layerNorm import LayerNorm

class GPT(torch.nn.Module):
    """
    GPT model that uses multi-head attention for language modeling.
    It consists of multiple transformer blocks, each containing a multi-head attention layer and a feed-forward network.
    The model is designed to predict the next token in a sequence based on the previous tokens.
    Args:
        config (dict): Configuration dictionary containing model parameters
    """
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = nn.Embedding(config['context_length'], config['emb_dim'])
        self.drop_emb = nn.Dropout(config['drop_rate'])

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config['n_layers'])
        ])

        self.final_ln = LayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

    def forward(self, in_idx):
        """
        Forward pass of the GPT model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        Returns:
            torch.Tensor: Output logits of shape (batch_size, sequence_length, vocab_size).
        """
        batch_size, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        x = tok_emb + pos_emb
        x = self.drop_emb(x)

        x = self.transformer_blocks(x)
        x = self.final_ln(x)
        logits = self.out_head(x)

        return logits