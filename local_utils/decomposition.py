from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from local_models.KronLinear import KronLinear
import numpy as np
import tensorly as tl

# def kron(a, b, s=None):
#     """

#     Args:
#         a (_type_): _description_
#         b (_type_): _description_
#         s (_type_, optional): _description_. Defaults to None.
#     """
#     # kronecker for a and b
#     # a have shape (rank, *a_shape)
#     # b have shape (rank, *b_shape)
#     # s have shape (*a_shape)
#     # if s is not None, then a will be multiplied by s
    
#     if s is not None:
#         assert a.shape[1:] == s.shape, "a and s should have the same shape"
#         a = s.unsqueeze(0) * a
    

def multidimensional_unfold(tensor: torch.Tensor, kernel_size: tuple, stride: tuple,
                            device: torch.device = torch.device('cpu')) -> torch.Tensor:
    r"""Unfolds `tensor` by extracting patches of shape `kernel_size`.

    Reshaping and traversal for patch extraction both follow C-order convention (last index changes the fastest).

    Args:
        tensor: Input tensor to be unfolded with shape [N, *spatial_dims] (N is batch dimension)
        kernel_size: Patch size.
        stride: Stride of multidimensional traversal.
        device: Device used for operations.

    Returns:
       Unfolded tensor with shape [N, :math:`\prod_k kernel_size[k]`, L]

    """

    s_dims = tensor.shape[1:]  # spatial dimensions

    # Number of positions along each axis
    num_positions = [np.floor((s_dims[i] - (kernel_size[i] - 1) - 1) / stride[i] + 1).astype(int)
                     for i in range(len(s_dims))]

    # Start indices for each position in each axis
    positions = [torch.tensor([n * stride[i] for n in range(num_positions[i] - 1, -1, -1)]) for i in
                 range(len(num_positions))]

    # Each column is a flattened patch
    output = torch.zeros(tensor.size(0), np.prod(kernel_size).item(), np.prod(num_positions).item(), device=device)

    for i, pos in enumerate(torch.cartesian_prod(*positions)):
        start_pos = torch.tensor([0, *pos])
        end_pos = torch.tensor([tensor.size(0), *(pos + torch.tensor(kernel_size))])
        patch = multidimensional_slice(tensor, start_pos, end_pos)  # n,f2,c2,h2,w2
        output[:, :, np.prod(num_positions).item() - 1 - i] = patch.reshape(tensor.size(0), -1)

    return output


def multidimensional_slice(tensor: torch.Tensor, start: torch.Tensor, stop: torch.Tensor) -> torch.Tensor:
    """Returns A[start_1:stop_1, ..., start_n:stop_n] for tensor A"

    Args:
        tensor: Input tensor `A`
        start: start indices
        stop: stop indices

    Returns:
         A[start_1:stop_1, ..., start_n:stop_n]
    """
    slices = [slice(start[i], stop[i]) for i in range(len(start))]
    return tensor[slices]



def kron(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray], s: Union[torch.Tensor, np.ndarray]=None) -> torch.Tensor:
    """Kronecker product between factors `a` and `b`

    Args:
        a: First factor
        b: Second factor

    Returns:
        Tensor containing kronecker product between `a` and `b`
    """
    if s is not None:
        assert a.shape[1:] == s.shape, "a and s should have the same shape"
        a = s.unsqueeze(0) * a
    a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    b = torch.from_numpy(b) if isinstance(b, np.ndarray) else b

    return torch.stack([torch.kron(a[k], b[k]) for k in range(a.shape[0])]).sum(dim=0)

def decompose_model(model, type, config):
    
    
    pass

def kron_decompose_model(model, layer_config=None):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = kron_decompose_model(module, layer_config)
            
        elif isinstance(module, nn.Conv2d):
            # todo: do the decomposition for nn.Conv2d
            pass
        elif isinstance(module, nn.Linear):
            linear_layer = module
            decomposed = linear2kronlinear(linear_layer, config=layer_config)
            model._modules[name] = decomposed
    return model

def kron_ensemble_model(model, layer_config=None):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = kron_ensemble_model(module, layer_config=layer_config)
            
        elif isinstance(module, KronLinear):
            ensemble = kronlinear2linear(module, config=layer_config)
            model._modules[name] = ensemble
    

def svd_decomposed_linear_model(linear_layer, rank, config):
    # todo : get the config settings
    # Weight should be size of (in_features, out_features)
    # A should have a size of (in_1, out_1)
    # B should have a size of (in_2, out_2)
    # input should have a size of (in_features)
    in_1, out_1 = config['a_shape']
    in_2, out_2 = config['b_shape']
    
    first_layer = nn.Linear(in_features = in_1, out_features = out_1, bias=False)
    second_layer = nn.Linear(in_features = in_1, out_features = out_2, bias=False)
    
    
    
    
    
    
def linear2kronlinear(linear_layer, rank=None, config=None):
    rank_rate = 0.5
    if config is not None:
        rank_rate = config['rank_rate'] if 'rank_rate' in config else 1
        shape_bias = config['shape_bias'] if 'shape_bias' in config else 0
        rank = config['rank'] if 'rank' in config else 0
    kronlinear = KronLinear(linear_layer.weight.shape[1], linear_layer.weight.shape[0], rank_rate=rank_rate, structured_sparse=True, bias=True, shape_bias=shape_bias, rank=rank)
    
    return kronlinear
    
    
    
def kronlinear2linear(kronlinear_layer, config=None):
    from gkpd.tensorops import kron
    a = kronlinear_layer.a * kronlinear_layer.s
    b = kronlinear_layer.b
    weight = kron(a, b)
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
    linear.weight = nn.Parameter(weight)
    linear.bias = kronlinear_layer.bias
    return linear
            

def freeze_A(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_A(module)
        else:
            if isinstance(module, KronLinear):
                module.a.requires_grad = False
            else:
                continue
    
def unfreeze_A(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_A(module)
        else:
            if isinstance(module, KronLinear):
                module.a.requires_grad = True
            else:
                continue
            
def freeze_B(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_B(module)
        else:
            if isinstance(module, KronLinear):
                module.b.requires_grad = False
            else:
                continue
            
def unfreeze_B(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_B(module)
        else:
            if isinstance(module, KronLinear):
                module.b.requires_grad = True
            else:
                continue
            
def freeze_S(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_A(module)
        else:
            if isinstance(module, KronLinear):
                module.s.requires_grad = False
            else:
                continue
            
def unfreeze_S(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_A(module)
        else:
            if isinstance(module, KronLinear):
                module.s.requires_grad = True
            else:
                continue



def fasterKDP(x, a, b):
    x_shape = x.shape
    a_shape = a.shape
    b_shape = b.shape
    
    
    assert a_shape[0] * b_shape[0] == x_shape[-1], "The shapes of the input tensor and the factors are not compatible"
    # change x[-1] into a[0] b[0]
    x = x.view(-1, a_shape[0], b_shape[0])    
    x = x @ b
    x = torch.permute(x, (0, 2, 1)).contiguous()
    x = x @ a
    x = torch.permute(x, (0, 2, 1)).contiguous()
    
    y_shape = [*x_shape[:-1], a_shape[-1] * b_shape[-1]]
    x = x.view(y_shape)
    return x
    
    
            
            
            
            
            
            
            
            
            