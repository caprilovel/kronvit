import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from deepspeed.profiling.flops_profiler import get_model_profile

print(os.environ['RANK'])
from timm.models import create_model


model = create_model('kron_deit_tiny_patch16_224', pretrained=True, kron_rank=10, shape_bias=3, freeze_A=True)

get_model_profile(model, input_shape=(1, 3, 224, 224), print_profile=True)