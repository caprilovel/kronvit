import torch
from local_utils.decomposition import kronecker_product_decompose, kron, low_rank_approximation




if __name__ == "__main__":
    w = torch.randn(40, 120)
    a_shape = (5, 12)
    b_shape = (8, 10)
    
    
    for i in range(1, 40, 5):
        a, b = kronecker_product_decompose(w, a_shape, b_shape, rank=i)
        w_1  = kron(a, b)
        
        l, r = low_rank_approximation(w, i)
        w_2 = l @ r
        print(torch.dist(w, w_1), torch.dist(w, w_2))