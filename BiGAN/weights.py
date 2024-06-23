import torch 


def init_weights(layer):
    name = layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(layer.weight, mean=0, std=0.05)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)
    elif name == 'Conv2d':
        torch.nn.init.normal_(layer.weight, mean=0, std=0.02)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)