import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization layer that normalizes the input across the last dimension.
    This is typically used in transformer architectures to stabilize training and improve convergence.
    
    Args:
        d_in (int): Dimension of the input features.
    """
    def __init__(self, d_in):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(d_in)) 
        self.shift = nn.Parameter(torch.zeros(d_in))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.scale + self.shift