import torch
import torch.nn as nn

class RELU(nn.Module):
    """
    Rectified Linear Unit (ReLU) activation function.
    This is a simple activation function that outputs the input directly if it is positive; otherwise, it will output zero.
    It is commonly used in neural networks to introduce non-linearity.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(0, x)

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    This activation function is a smooth approximation of the ReLU function and is often used in transformer models.
    It applies a non-linear transformation based on the Gaussian distribution.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))