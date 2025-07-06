import torch
import torch.nn as nn
from utils.activation import GELU, RELU

class FeedForward(nn.Module):
    """
    FeedForward network that applies a linear transformation followed by an activation function.
    This is typically used in transformer architectures to process the output of attention layers.
    
    Args:
        embed_dim (int): Embedding dimension size.
        activation (str): Activation function to use ('relu' or 'gelu').
    """
    def __init__(self, embed_dim, activation='relu'):
        super().__init__()

        if activation == 'relu':
            self.activation = RELU()
        elif activation == 'gelu':
            self.activation = GELU()
        else:
            raise ValueError("Unsupported activation function. Use 'relu' or 'gelu'.")
        
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # First linear layer to expand the dimension
            self.activation,  # Apply activation function
            nn.Linear(4*embed_dim, embed_dim)  # Final linear layer to transform to output dimension
        )

    def forward(self, x):
        return self.layers(x)  # Apply linear transformation and activation