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
        d_out (int): Output dimension size.
        d_in (int): Input dimension size.
        context_len (int): Length of the context for causal attention.
        dropout (float): Dropout rate for attention weights.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to use bias in the linear transformations.
    """
    def __init__(self, d_out, d_in, context_len, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert( d_out % num_heads == 0 ), "Output dimension must be divisible by number of heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out, bias=qkv_bias) # Linear layer to concatenate head outputs
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))
        assert( d_out % num_heads == 0 ), "Output dimension must be divisible by number of heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out, bias=qkv_bias) # Linear layer to concatenate head outputs
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        b, num_tokens, d_in = x.shape

        # reshape keys, queries, values to (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # Contatenate head outputs
        
        # Apply final linear transformation to the concatenated context vector
        context_vec = self.out_proj(context_vec)
        return context_vec
