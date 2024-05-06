from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from local_models.KronLinear import KronLinear
import numpy as np
import tensorly as tl



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



def gkpd(tensor: torch.Tensor, a_shape: Union[list, tuple], b_shape: Union[list, tuple],
         atol: float = 1e-3) -> tuple:
    """Finds Kronecker decomposition of `tensor` via SVD.
    Patch traversal and reshaping operations all follow a C-order convention (last dimension changing fastest).
    Args:
        tensor (torch.Tensor): Tensor to be decomposed.
        a_shape (list, tuple): Shape of first Kronecker factor.
        b_shape (list, tuple): Shape of second Kronecker factor.
        atol (float): Tolerance for determining tensor rank.

    Returns:
        a_hat: [rank, *a_shape]
        b_hat: [rank, *b_shape]
    """

    if not np.all(np.array([a_shape, b_shape]).prod(axis=0) == np.array(tensor.shape)):
        raise ValueError("Received invalid factorization dimensions for tensor during its GKPD decomposition")

    with torch.no_grad():
        w_unf = multidimensional_unfold(
            tensor.unsqueeze(0), kernel_size=b_shape, stride=b_shape
        )[0].T  # [num_positions, prod(s_dims)]

        u, s, v = torch.svd(w_unf)
        rank = len(s.detach().numpy()[np.abs(s.detach().numpy()) > atol])

        # Note: pytorch reshaping follows C-order as well
        a_hat = torch.stack([s[i].item() * u[:, i].reshape(*a_shape) for i in range(rank)])  # [rank, *a_shape]
        b_hat = torch.stack([v.T[i].reshape(*b_shape) for i in range(rank)])  # [rank, *b_shape]

    return a_hat, b_hat

def fasterKDP(x, a, b):
    """faster calculation of Kronecker product, assume that W = A ⊗ B, and x is the input, then this function will reshape
    x , A and B, make the calculation of W @ x more efficient.

    Args:
        x (torch.tensor): input tensor, shape is [batch_size, ..., a_shape[0] * b_shape[0]]
        a (torch.tensor): factor a, shape is [a_shape[0], a_shape[1]]
        b (torch.tensor): factor b, shape is [b_shape[0], b_shape[1]]

    Returns:
        output (torch.tensor): output tensor, shape is [batch_size, ..., a_shape[1] * b_shape[1]]
    """
    x_shape = x.shape
    a_shape = a.shape
    b_shape = b.shape
    
    if len(a_shape) == 2 and len(b_shape) == 2:        
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
    
    elif len(a_shape) == 3 and len(b_shape) == 3:
        assert a_shape[1] * b_shape[1] == x_shape[-1], "The shapes of the input tensor and the factors are not compatible"
        assert a_shape[0] == b_shape[0], "The ranks of the factors are not compatible"
        # change x[-1] into a[0] b[0]
        x = x.view(-1, a_shape[1], b_shape[1])    
        x = x @ b
        x = torch.permute(x, (0, 2, 1)).contiguous()
        x = x @ a
        x = torch.permute(x, (0, 2, 1)).contiguous()
        
        y_shape = [*x_shape[:-1], a_shape[-1] * b_shape[-1]]
        x = x.view(y_shape)
        return x



def low_rank_approximation(weight, rank):
    U, S, V = torch.linalg.svd(weight)
    u = U[:, :rank]
    v = V[:rank, :]
    s = torch.sqrt(S[:rank])
    u = u @ torch.diag(s)
    v = torch.diag(s) @ v
    return u, v
    

    

    
    
            
            
            
            
            
            
            
            
            