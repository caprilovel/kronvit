#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.kron_vision_transformer import KronLinear


kl = KronLinear(1000, 1000)
linear = nn.Linear(1000, 1000)

input = torch.randn(100, 100, 1000)
kl, linear = kl.cuda(), linear.cuda()
input = input.cuda()
#%% 
# calculate the time for kron linear
times = 100
kl_start_time = torch.cuda.Event(enable_timing=True)

kl_end_time = torch.cuda.Event(enable_timing=True)
kl_start_time.record()
for i in range(times):
    output = kl(input)
kl_end_time.record()
torch.cuda.synchronize()
kl_elapsed_time = kl_start_time.elapsed_time(kl_end_time)
print(f"kron linear elapsed time: {kl_elapsed_time}")



# calculate the time for linear
linear_start_time = torch.cuda.Event(enable_timing=True)
linear_end_time = torch.cuda.Event(enable_timing=True)
linear_start_time.record()
for i in range(times):
    output = linear(input)
linear_end_time.record()
torch.cuda.synchronize()
linear_elapsed_time = linear_start_time.elapsed_time(linear_end_time)
print(f"linear elapsed time: {linear_elapsed_time}")

