def model_parameter_count(model):
    """
    Calculate the total number of trainable parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model for which to count parameters.
        
    Returns:
        int: Total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)