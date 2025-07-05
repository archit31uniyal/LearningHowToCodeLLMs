import torch
import math

class SelfAttention(torch.nn.Module):
    """Self-Attention mechanism that computes attention scores between input vectors.
    It uses three linear transformations for queries, keys, and values.
    The attention scores are computed as the dot product of queries and keys,followed by a softmax operation to obtain attention weights.
    The context vector is then computed as the weighted sum of values based on these attention weights.
    
    Args:
        d_in (int): Input dimension size.
        d_out (int): Output dimension size.
        qkv_bias (bool): Whether to use bias in the linear transformations.
    """
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attention_weights = torch.nn.Softmax(attn_scores / math.sqrt(keys.shape[-1]), dim=-1)
        context_vec = attention_weights @ values
        return context_vec

class CausalAttention(torch.nn.Module):
    """
    Causal Attention mechanism that ensures the model can only attend to previous tokens.
    """
    def __init__(self, d_in, d_out, context_len, dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf())
        attention_weights = torch.nn.Softmax(attn_scores / math.sqrt(self.W_key.shape[-1]), dim=-1)
        attention_weights = self.Dropout(attention_weights)
        context_vec = attention_weights @ values
        return context_vec

class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention mechanism that applies multiple self-attention heads in parallel.
    Each head computes its own attention scores and context vectors, which are then concatenated.
    Args:
        embed_size (int): Size of the input embeddings.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to use bias in the linear transformations.
    """
    def __init__(self, d_out, d_in, context_len, dropout, num_heads, qkv_bias = False):
        super().__init__()

        self.heads = torch.nn.ModuleList(
            [CausalAttention(d_in, d_out, context_len, dropout, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.concat([head(x) for head in self.heads], dim=-1)