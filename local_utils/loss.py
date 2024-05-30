import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

import numpy as np






# -----------------------------------------------
# for Linear Layer
# -----------------------------------------------


def group_pattern(n: int, m: int, mat: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """will group the matrix into n x m blocks
    
    For example:
    if  W is a matrix of size (8, 6), n = 2, m = 3
    
    then the matrix will be grouped into (4,2) blocks of size (2, 3)
    [[G1, G2],
        [G3, G4],
        [G5, G6],
        [G7, G8]]
    
    Gi = [[W[2i, 3j], W[2i, 3j+1], W[2i, 3j+2]],

    Args:
        n (int): the 
        m (int): _description_
        mat (Union[torch.Tensor, np.ndarray]): _description_

    Returns:
        torch.Tensor: _description_
    """
    mat_shape = mat.shape
    assert len(mat_shape) == 2, "The input matrix should be 2D"
    assert mat_shape[0] % n == 0 and mat_shape[1] % m == 0, "The input matrix should be divisible by n and m"
    n1 = mat_shape[0] // n
    m1 = mat_shape[1] // m
    from einops import rearrange
    mat = rearrange(mat, '(n1 n) (m1 m) -> (n1 m1) (n m)', n=n, m=m, n1=n1, m1=m1)
    return mat

def group_lasso(model, pattern, lr=0.01):
    reg_loss = 0
    total_params = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            reg_loss += group_lasso(module, pattern, lr=lr)[0]
            total_params += group_lasso(module, pattern, lr=lr)[1]
        elif isinstance(module, nn.Linear):
            norm = torch.norm(group_pattern(pattern[0], pattern[1], module.weight), p=2, dim=1)
            reg_loss += norm.sum() / (np.sqrt(pattern[0] * pattern[1]) * norm.numel())
            total_params += 1
    return reg_loss, total_params

def l1_linear_norm(model):
    reg_loss = 0
    total_params = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            reg_loss += l1_linear_norm(module)[0]
            total_params += l1_linear_norm(module)[1]
        elif isinstance(module, nn.Linear):
            reg_loss += module.weight.abs().sum()
            total_params += module.weight.numel()
    return reg_loss, total_params

def elastic_group_lasso(model, pattern, lr=0.1, alpha=0.05):
    return lr * group_lasso(model, pattern, lr=lr)[0] / group_lasso(model, pattern, lr=lr)[1] + alpha * l1_linear_norm(model)[0] / l1_linear_norm(model)[1]


def linear_weight_l1_norm(model):
    reg_loss = 0
    total_params = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            reg_loss += linear_weight_l1_norm(module)[0]
            total_params += linear_weight_l1_norm(module)[1]
        elif isinstance(module, nn.Linear):
            reg_loss += module.weight.abs().sum()
            total_params += module.weight.numel()
    return reg_loss, total_params

def sparse_linear(model, thresold=1e-4):
    s_total_num = 0
    zero_num = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            s_total_num += sparse_linear(module, thresold=thresold)[0]
            zero_num += sparse_linear(module, thresold=thresold)[1]
        elif isinstance(module, nn.Linear):
            s_total_num += module.weight.numel()
            zero_num += torch.abs(module.weight)[module.weight < thresold].numel()
    return s_total_num, zero_num
        
# -----------------------------------------------
# for Kronecker Product Decomposition Linear Layer
# ----------------------------------------------- 
from local_models.KronLinear import KronLinear

def norm_s(model, lr=0.01, p=1):
    reg_loss = 0
    total_params = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            reg_loss += norm_s(module, lr=lr, p=p)[0]
            total_params += norm_s(module, lr=lr, p=p)[1]
        elif isinstance(module, KronLinear):
            reg_loss += torch.norm(module.s, p=p)
            total_params += module.s.numel()
    return reg_loss, total_params

def sparse_s(model, thresold=1e-4):
    s_total_num = 0
    zero_num = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            s_total_num += sparse_s(module, thresold=thresold)[0]
            zero_num += sparse_s(module, thresold=thresold)[1]
        elif isinstance(module, KronLinear):
            s_total_num += module.s.numel()
            zero_num += torch.abs(module.s)[module.s < thresold].numel()
    return s_total_num, zero_num