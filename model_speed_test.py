#%%
import torch
import torch.nn as nn

from timm.models import vision_transformer, kron_vision_transformer
from timm.models.vision_transformer import VisionTransformer
from timm.models.kron_vision_transformer import KronVisionTransformer
from functools import partial
# %%

model1 = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), )
optimizer1 = optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)

model2 = KronVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), )
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

input = torch.randn(100, 3, 224, 224)
input_shape = (100, 3, 224, 224)
model1, model2 = model1.cuda(), model2.cuda()
input = input.cuda()
#%%
times = 1

model1_start_time = torch.cuda.Event(enable_timing=True)
model1_training_time = torch.cuda.Event(enable_timing=True)
model1_end_time = torch.cuda.Event(enable_timing=True)
model1_start_time.record()
for i in range(times):
    output1 = model1(input)
    model1_training_time.record()
    optimizer1.zero_grad()
    output1.mean().backward()
    optimizer1.step()
model1_end_time.record()
torch.cuda.synchronize()
model1_elapsed_time = model1_start_time.elapsed_time(model1_end_time)
model1_training_time = model1_start_time.elapsed_time(model1_training_time)
print(f"model1 training time: {model1_training_time}")
print(f"model1 elapsed time: {model1_elapsed_time}")

model2_start_time = torch.cuda.Event(enable_timing=True)
model2_end_time = torch.cuda.Event(enable_timing=True)
model2_training_time = torch.cuda.Event(enable_timing=True)
model2_start_time.record()
for i in range(times):
    output2 = model2(input)
    model2_training_time.record()
    optimizer2.zero_grad()
    output2.mean().backward()
    optimizer2.step()
model2_end_time.record()
torch.cuda.synchronize()
model2_elapsed_time = model2_start_time.elapsed_time(model2_end_time)
model2_training_time = model2_start_time.elapsed_time(model2_training_time)
print(f"model2 elapsed time: {model2_elapsed_time}")
print(f"model2 training time: {model2_training_time}")
#%%
# calcu the params of the model
from torchinfo import summary
# summary(model1, input_shape)
# summary(model2, input_shape)

model1_layer_names = [name for name, _ in model2.named_parameters()] 
# delete the name which contains at least 2 '.'
model1_layer_names = [name for name in model1_layer_names if name.count('.') < 3]



model2_layer_names = [name for name, _ in model2.named_parameters()]
# print(model2_layer_names)


# %%
import time 
take_time_dict = {}
def take_time_pre(layer_name, module, input):
    take_time_dict[layer_name] = time.time() 

def take_time(layer_name,module, input, output):
    take_time_dict[layer_name] =  time.time() - take_time_dict[layer_name]
    
for layer in model2.children():
    layer.register_forward_pre_hook( partial(take_time_pre, layer) )
    layer.register_forward_hook( partial(take_time, layer) )
#%%
output2 = model2(input)
# print(take_time_dict)