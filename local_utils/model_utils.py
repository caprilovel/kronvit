from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

from local_models.KronLinear import Kron1Linear, VeKronLinear


def k1l_decompose_model(model, layer_config=None):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = k1l_decompose_model(module, layer_config)
            
        elif isinstance(module, nn.Conv2d):
            # todo: do the decomposition for nn.Conv2d
            pass
        elif isinstance(module, nn.Linear):
            linear_layer = module
            decomposed = linear2k1l(linear_layer, config=layer_config)
            model._modules[name] = decomposed
    return model



def linear2k1l(linear_layer, config=None):
    if config is not None:
        shape_bias = config['shape_bias'] if 'shape_bias' in config else 0
        structured_sparse = config['structured_sparse'] if 'structured_sparse' in config else False
        print(config['structured_sparse'])
    kronlinear = Kron1Linear(linear_layer.weight.shape[1], linear_layer.weight.shape[0], structured_sparse=structured_sparse, shape_bias=shape_bias)
    
    return kronlinear


def k1l_update_model(model, layer_config=None):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = k1l_update_model(module, layer_config)
            
        elif isinstance(module, Kron1Linear):
            module.update_W_0()
    return model

def k1l_freezes(model):
    total_params = 0
    zero_params = 0
    #calcu the total params and zero params of s

    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            out = k1l_freezes(module)
            total_params += out[0]
            zero_params += out[1]
            
        elif isinstance(module, Kron1Linear):
            module.freeze_S()
            # return s numels and zeros numels(zero <1e-5)
            # return module.s.numel(), module.s[module.s<1e-5].numel()
            
    return total_params, zero_params


def s_l1_loss(model):
    loss = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            loss += s_l1_loss(module)
            
        elif isinstance(module, Kron1Linear):
            loss += module.s.abs().sum()
    return loss

def s_l2_loss(model):
    loss = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            loss += s_l2_loss(module)
            
        elif isinstance(module, Kron1Linear):
            loss += module.s.pow(2).sum()
    return loss



#####

def linear2vekronlinear(linear_layer, config=None):
    if config is not None:
        shape_bias = config['shape_bias'] if 'shape_bias' in config else 0
        structured_sparse = config['structured_sparse'] if 'structured_sparse' in config else False
        print(config['structured_sparse'])
    vekronlinear = VeKronLinear(linear_layer.weight.shape[1], linear_layer.weight.shape[0], structured_sparse=structured_sparse, shape_bias=shape_bias, rank=1000)
    
    return vekronlinear 

def vekron_decompose_model(model, layer_config=None):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = vekron_decompose_model(module, layer_config)
            
        elif isinstance(module, nn.Conv2d):
            # todo: do the decomposition for nn.Conv2d
            pass
        elif isinstance(module, nn.Linear):
            linear_layer = module
            decomposed = linear2vekronlinear(linear_layer, config=layer_config)
            model._modules[name] = decomposed
    return model