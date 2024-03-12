import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
# from gkpd.tensorops import kron
# from utils.factorize import factorize
from typing import Optional
from einops import rearrange

from torch.jit import Final


from typing import List


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

class KronLinear(nn.Module):
    def __init__(self, in_features, out_features, shape_bias=0, structured_sparse=False, bias=True, rank_rate=0.1, rank=0) -> None:
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
        self.rank = rank if rank > 0 else min(a_shape[0], a_shape[1], b_shape[0], b_shape[1]) * rank_rate
        self.rank = int(self.rank) if int(self.rank) > 0 else 1
        
        self.structured_sparse = structured_sparse
        
        if structured_sparse:
            self.s = nn.Parameter(torch.randn( *a_shape), requires_grad=True)
        else:
            self.s = None
        self.a = nn.Parameter(torch.randn(self.rank, *a_shape), requires_grad=True)
        self.b = nn.Parameter(torch.randn(self.rank, *b_shape), requires_grad=True)
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
        # a = self.a
        # if self.structured_sparse:
        #     a = self.s.unsqueeze(0) * self.a
        
        # # a = self.s.unsqueeze(0) * self.a
        # w = kron(a, self.b)
        
        # out = x @ w 
        # if self.bias is not None:
        #     out += self.bias.unsqueeze(0)
        # return out
        # =========================
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

        # Permute dimensions
        out = out.permute(0, 2, 1).contiguous()

        # Reshape again
        out = out.view(-1, self.a_shape[2] * self.b_shape[2])
        
        
        
        out = torch.reshape(out, x_shape[:-1] + (self.a_shape[2] * self.b_shape[2],))
        
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out
    
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


if __name__ == "__main__":
    from torchsummary import summary
    lenet5 = KronLeNet_5()
    a = torch.randn(1, 1, 28, 28)
    summary(lenet5, (1, 28, 28), device='cpu')
    # CALCULATE PARAMS
    params = 0
    for name, param in lenet5.named_parameters():
        print(name, param.shape)
        params += param.numel()
    print(params)
    
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