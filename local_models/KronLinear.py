import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
# from gkpd.tensorops import kron
# from utils.factorize import factorize
from typing import Optional, Union
from einops import rearrange

from torch.jit import Final


from typing import List

def kron(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Kronecker product between factors `a` and `b`

    Args:
        a: First factor
        b: Second factor

    Returns:
        Tensor containing kronecker product between `a` and `b`
    """

    a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    b = torch.from_numpy(b) if isinstance(b, np.ndarray) else b

    return torch.stack([torch.kron(a[k], b[k]) for k in range(a.shape[0])]).sum(dim=0)



def fasterKDP(x, a, b):
    x_shape = x.shape
    a_shape = a.shape
    b_shape = b.shape
    
    assert len(a_shape) == len(b_shape) , "The shapes of the factors are not compatible"
    if len(a_shape) == 2:
        assert a_shape[0] * b_shape[0] == x_shape[-1], "The shapes of the input tensor and the factors are not compatible"
    elif len(a_shape) == 3:
        assert a_shape[1] * b_shape[1] == x_shape[-1], "The shapes of the input tensor and the factors are not compatible"
    
    if len(a_shape) == 2:
        x = x.view(-1, a_shape[0], b_shape[0])    
        x = x @ b
        x = torch.permute(x, (0, 2, 1)).contiguous()
        x = x @ a
        x = torch.permute(x, (0, 2, 1)).contiguous()
        
        y_shape = [*x_shape[:-1], a_shape[-1] * b_shape[-1]]
        x = x.view(y_shape)
        return x
    
    elif len(a_shape) == 3:
        w = kron(a, b)
        return x @ w




def factorize(n: int, bias=0) -> List[int]:
    flag = False
    if bias <0:
        flag = True
    # """Return the most average two factorization of n."""
    for i in range(int(np.sqrt(n)) + 1, 1, -1):
        if n % i == 0:
            if bias == 0:
                return [i, n // i] if not flag else [n // i, i]
            else:
                bias -= 1
    return [n, 1] if not flag else [1, n]


Kronnecker_group = [[
    [(16, 10 * 5), (16, 12 * 2)],
    [(10 * 5, 12 * 2), (12 * 2, 7 * 5)],
    [(12 * 2, 2), (7 * 5, 5)]
    ],
    [[(8, 10), (32, 12)],
     [(5, 6), (24, 14)],
     [(4, 2), (21, 5)]
     ],
    [[(32, 12), (8, 10)],
     [(24, 14), (5, 6)],
        [(21, 5), (4, 2)]
     ],
    [[(4, 5), (64, 24)],
     [(5, 3), (24, 28)],
     [(4, 2), (21, 5)]
     ],
    ]

class Kron1Linear(nn.Module):
    def __init__(self, in_features, out_features, shape_bias, structured_sparse=True, *args, **kwargs) -> None:
        super().__init__()
        
        
        in_shape = factorize(in_features, shape_bias)
        out_shape = factorize(out_features, shape_bias)
        a_shape = (in_shape[0], out_shape[1])
        b_shape = (in_shape[1], out_shape[0])# change the order of the shape
        self.structured_sparse = structured_sparse
        
        if structured_sparse:
            self.s = nn.Parameter(torch.randn( *a_shape), requires_grad=True)
        
        self.a = nn.Parameter(torch.randn(*a_shape), requires_grad=True)
        self.b = nn.Parameter(torch.randn(*b_shape), requires_grad=True)
        
        self.W_0 = nn.Parameter(torch.zeros(in_features, out_features), requires_grad=False)
        
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
    
    def forward(self, x):
        a = self.a
        if self.structured_sparse:
            a = self.s * self.a
        y_1 = x @ self.W_0
        y_2  = fasterKDP(x, a, self.b)
        y = y_1 + y_2
        return y

    def update_W_0(self):
        with torch.no_grad():
            if self.structured_sparse:
                self.W_0 += torch.kron((self.s * self.a), self.b)
            self.W_0 += torch.kron(self.a, self.b)
        # nn.init.xavier_uniform_(self.a)
        # nn.init.xavier_uniform_(self.b)
        self.a.data = torch.zeros_like(self.a)
        self.b.data = nn.init.xavier_uniform_(self.b)
        
    def freeze_S(self, flag=False):
        # set the abs less than 1e-6 to 0 and freeze s
        if self.structured_sparse:
            self.s.data = torch.where(torch.abs(self.s) < 1e-6, torch.zeros_like(self.s), self.s)
            self.s.requires_grad = flag
        
    def random_mask_S(self, zero_rate=0.5):
        zero_map = torch.bernoulli(torch.ones_like(self.s) * zero_rate)
        if self.structured_sparse:
            self.s = torch.where(zero_map == 0, self.s, torch.zeros_like(self.s))
            # self.s = torch.where(zero_map == 0, self.s, torch.zeros_like(self.s))
            self.s.requires_grad = False
        
class VeKronLinear(nn.Module):
    def __init__(self, in_features, out_features, shape_bias=0, structured_sparse=False, bias=True, rank=256) -> None:
        """Kronecker Linear Layer

        Args:
            rank (int): _description_
            a_shape (_type_): _description_
            b_shape (_type_): _description_
            structured_sparse (bool, optional): _description_. Defaults to False.
            bias (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        
        in_shape = factorize(in_features, shape_bias)
        out_shape = factorize(out_features, shape_bias)
        a_shape = (in_shape[0], out_shape[1])
        b_shape = (in_shape[1], out_shape[0])# change the order of the shape

        self.rank = rank if rank > 0 else 1
        
        self.structured_sparse = structured_sparse
        
        if structured_sparse:
            self.s = nn.Parameter(torch.randn( *a_shape), requires_grad=True)
        else:
            self.s = None
        self.a = nn.Parameter(torch.randn(self.rank, *a_shape), requires_grad=False)
        self.b = nn.Parameter(torch.randn(self.rank, *b_shape), requires_grad=False)
        
        self.a_lambda = nn.Parameter(torch.randn(8, self.rank), requires_grad=True)
        self.b_lambda = nn.Parameter(torch.randn(8, self.rank), requires_grad=True)
        
        self.a_lambda.data = torch.ones_like(self.a_lambda)
        self.b_lambda.data = torch.zeros_like(self.b_lambda)
        
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
        self.a_shape = self.a.shape
        self.b_shape = self.b.shape
        bias_shape = np.multiply(a_shape, b_shape)
        if bias:
            self.bias = nn.Parameter(torch.randn(*bias_shape[1:]), requires_grad=True)
        else:
            self.bias = None
        print(self.a_shape, self.b_shape)
        
    def forward(self, x):
        a = self.a
        if self.structured_sparse:
            a = self.s.unsqueeze(0) * self.a
        a = a.view(self.rank, -1)
        b = self.b.view(self.rank, -1)
        a = self.a_lambda @ a
        b = self.b_lambda @ b 
        r = self.a_lambda.shape[0]
        if r > 1:
            a = a.view(r, *self.a_shape[1:])
            b = b.view(r, *self.b_shape[1:])
        else:            
            a = a.view(*self.a_shape[1:])
            b = b.view(*self.b_shape[1:])
        
        out = fasterKDP(x, a, b) 
        
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out
    
class VeKronLinearRank1(nn.Module):
    def __init__(self, in_features, out_features, shape_bias=0, structured_sparse=False, bias=True, rank=256) -> None:
        """Kronecker Linear Layer

        Args:
            rank (int): _description_
            a_shape (_type_): _description_
            b_shape (_type_): _description_
            structured_sparse (bool, optional): _description_. Defaults to False.
            bias (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        
        in_shape = factorize(in_features, shape_bias)
        out_shape = factorize(out_features, shape_bias)
        a_shape = (in_shape[0], out_shape[1])
        b_shape = (in_shape[1], out_shape[0])# change the order of the shape

        self.rank = rank if rank > 0 else 1
        
        self.structured_sparse = structured_sparse
        
        if structured_sparse:
            self.s = nn.Parameter(torch.randn( *a_shape), requires_grad=True)
        else:
            self.s = None
        self.a = nn.Parameter(torch.randn(self.rank, *a_shape), requires_grad=False)
        self.b = nn.Parameter(torch.randn(self.rank, *b_shape), requires_grad=False)
        self.W_0 = nn.Parameter(torch.zeros(in_features, out_features), requires_grad=False)
        
        self.a_lambda = nn.Parameter(torch.randn(1, self.rank), requires_grad=True)
        self.b_lambda = nn.Parameter(torch.randn(1, self.rank), requires_grad=True)
        
        self.a_lambda.data = torch.ones_like(self.a_lambda)
        self.b_lambda.data = torch.zeros_like(self.b_lambda)
        
        nn.init.uniform_(self.a, -1, 1)
        nn.init.uniform_(self.b, -1, 1)
        self.a_shape = self.a.shape
        self.b_shape = self.b.shape
        bias_shape = np.multiply(a_shape, b_shape)
        if bias:
            self.bias = nn.Parameter(torch.randn(*bias_shape[1:]), requires_grad=True)
        else:
            self.bias = None
        print(self.a_shape, self.b_shape)
        
    def forward(self, x):
        a = self.a
        if self.structured_sparse:
            a = self.s.unsqueeze(0) * self.a
        a = a.view(self.rank, -1)
        b = self.b.view(self.rank, -1)
        a = self.a_lambda @ a
        b = self.b_lambda @ b 
        r = self.a_lambda.shape[0]
        if r > 1:
            a = a.view(r, *self.a_shape[1:])
            b = b.view(r, *self.b_shape[1:])
        else:            
            a = a.view(*self.a_shape[1:])
            b = b.view(*self.b_shape[1:])
        
        out = fasterKDP(x, a, b) 
        out += x @ self.W_0
        
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out
    
    def update_W_0(self):
        with torch.no_grad():
            if self.structured_sparse:
                self.W_0 += torch.kron((self.s * self.a), self.b)
            self.W_0 += torch.kron(self.a, self.b)
        print('update W_0')
        nn.init.uniform_(self.a, -1, 1)
        nn.init.uniform_(self.b, -1, 1)
        self.a_lambda.data = torch.ones_like(self.a_lambda)
        self.b_lambda.data = torch.zeros_like(self.b_lambda)
    
class KronLinear(nn.Module):
    def __init__(self, in_features, out_features, patch_size=None, structured_sparse=False, bias=True, rank=0) -> None:
        """Kronecker Linear Layer

        Args:
            rank (int): _description_
            a_shape (_type_): _description_
            b_shape (_type_): _description_
            structured_sparse (bool, optional): _description_. Defaults to False.
            bias (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        if patch_size is None:
            in_shape = factorize(in_features)
            out_shape = factorize(out_features)
        else:
            in_shape = [patch_size[0], in_features // patch_size[0]]
            out_shape = [patch_size[1], out_features // patch_size[1]]
        a_shape = (in_shape[1], out_shape[1])
        b_shape = (in_shape[0], out_shape[0])# change the order of the shape
  
  
        self.rank = int(rank) if int(rank) > 0 else 1
        
        self.structured_sparse = structured_sparse
        
        if structured_sparse:
            self.s = nn.Parameter(torch.randn( *a_shape), requires_grad=True)
        else:
            self.s = None
        self.a = nn.Parameter(torch.randn(self.rank, *a_shape), requires_grad=True)
        self.b = nn.Parameter(torch.randn(self.rank, *b_shape), requires_grad=True)
        nn.init.uniform_(self.a, -1, 1)
        nn.init.uniform_(self.b, -1, 1)
        self.a_shape = self.a.shape
        self.b_shape = self.b.shape
        bias_shape = np.multiply(a_shape, b_shape)
        if bias:
            self.bias = nn.Parameter(torch.randn(*bias_shape[1:]), requires_grad=True)
        else:
            self.bias = None
        
        
    def forward(self, x):
        a = self.a
        if self.structured_sparse:
            a = self.s.unsqueeze(0) * self.a
        
        # a = self.s.unsqueeze(0) * self.a
        # w = kron(a, self.b)
        x_shape = x.shape 
        b = self.b
        
        x = torch.reshape(x, (-1, x_shape[-1]))
        
        # b = rearrange(b, 'r b1 b2 -> b1 (b2 r)')
        b = b.permute(1, 2, 0).contiguous().view(b.shape[1], -1).contiguous()
        
        # x = rearrange(x, 'n (a1 b1) -> n a1 b1', a1=self.a_shape[1], b1=self.b_shape[1])
        x = x.view(-1, self.a_shape[1], self.b_shape[1])
        out = x @ b
        
        # out = rearrange(out, 'n a1 (b2 r) -> r (n b2) a1', b2=self.b_shape[2], r=self.rank) 
        out = out.view(-1, self.a_shape[1], self.rank, self.b_shape[2])

        # Permute dimensions
        out = out.permute(2, 0, 3, 1)
        out = out.contiguous().view(self.rank, -1, self.a_shape[1])
        out = torch.bmm(out, a)
        out = torch.sum(out, dim=0).squeeze(0)
        
        
        # out = rearrange(out, '(n b2) a2 -> n (a2 b2)', b2=self.b_shape[2])
        out = out.view(-1, self.b_shape[2], self.a_shape[2])
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(-1, self.a_shape[2] * self.b_shape[2])
        
        
        
        out = torch.reshape(out, x_shape[:-1] + (self.a_shape[2] * self.b_shape[2],))
        
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out
    
    def generate_from_linear(self, linear_layer):
        weight = linear_layer.weight
        from local_utils.tensorops import gkpd
        a_shape = (self.a_shape[2], self.a_shape[1])
        b_shape = (self.b_shape[2], self.b_shape[1])
        rank = self.rank
        a, b = gkpd(weight, a_shape, b_shape)
        a = a[:rank]
        a = a.reshape(rank, a_shape[1], a_shape[0])
        b = b[:rank]
        b = b.reshape(rank, b_shape[1], b_shape[0])
        self.a = nn.Parameter(a, requires_grad=True)
        self.b = nn.Parameter(b, requires_grad=True)
        
    
class KronLeNet(nn.Module):
    def __init__(self, group_id=1) -> None:
        super(KronLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        rank1 = 15
        rank2 = 6
        rank3 = 3
        
        self.kronfc1 = KronLinear(rank1, Kronnecker_group[group_id][0][0], Kronnecker_group[group_id][0][1], bias=False, structured_sparse=True)
        
        self.kronfc2 = KronLinear(rank2, Kronnecker_group[group_id][1][0], Kronnecker_group[group_id][1][1], bias=False, structured_sparse=True)
        
        self.kronfc3 = KronLinear(rank3, Kronnecker_group[group_id][2][0], Kronnecker_group[group_id][2][1], bias=False, structured_sparse=True)
        self.relu3 = nn.LeakyReLU()
        self.relu4 = nn.LeakyReLU()


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu3(self.kronfc1(x))
        x = self.relu4(self.kronfc2(x))
        x = self.kronfc3(x)
        return x

"""
Kron Linear Config setting

rank_rate: to use the rate * min(a1, a2, b1, b2) as the rank of the kron linear layer
structured_sparse: whether to use the structured sparse
bias: whether to use the bias
shape_bias: the shape of the bias

"""

class KronLeNet_5(nn.Module):
    def __init__(self, kron_config=None) -> None:
        super().__init__()
        
        rank_rate = kron_config['rank_rate'] if kron_config  else 0.5
        structured_sparse = kron_config['structured_sparse'] if kron_config else True
        bias = kron_config['bias'] if kron_config else False
        shape_bias = kron_config['shape_bias'] if kron_config else 0
        
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.kronfc1 = KronLinear(25 * 4 * 4, 120, shape_bias=shape_bias, structured_sparse=structured_sparse, bias=bias, rank_rate=rank_rate)
        self.kronfc2 = KronLinear(120, 84, shape_bias=shape_bias, structured_sparse=structured_sparse, bias=bias, rank_rate=rank_rate)
        self.kronfc3 = KronLinear(84, 10, shape_bias=shape_bias, structured_sparse=structured_sparse, bias=bias, rank_rate=rank_rate)
        self.relu3 = nn.LeakyReLU()
        self.relu4 = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 25 * 4 * 4)
        x = self.relu3(self.kronfc1(x))
        x = self.relu4(self.kronfc2(x))
        x = self.kronfc3(x)
        return x


    
class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_rate=0.5, bias=True) -> None:
        super().__init__()
        self.rank = int(min(in_features, out_features) * rank_rate)
        self.rank = 1 if self.rank == 0 else self.rank
        self.a = nn.Parameter(torch.randn(self.rank, in_features), requires_grad=True)
        self.b = nn.Parameter(torch.randn(self.rank, out_features), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(out_features), requires_grad=True) if bias else None
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
    
    def forward(self, x):
        # x shape (batch, in_features)
        out = torch.mm(x, self.a.t())
        out = torch.mm(out, self.b)
        if self.bias is not None:
            out += self.bias
        return out
    


