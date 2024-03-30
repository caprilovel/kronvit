#!/bin/bash

# SCRIPT_PATH =  "/home/zhu.3723/kronvit/"
cd /home/zhu.3723/kronvit/
GPU_NUM=2
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3



start_time=$(date +"%s") 

torchrun --nproc_per_node=$GPU_NUM \
     --master_port 29503\
      main.py \
     --epochs 1500 \
     --model vekron_deit_tiny_patch16_224 \
     --batch-size 512 \
     --k1l \
     --data-set CIFAR \
     --data-path /local/storage/ding/cifar100 \
     --lr 5e-3\
     --warmup-lr 1e-3\
     --min-lr 1e-3\
     --output_dir /home/zhu.3723/kronvit/output/cifar100_vekron_bias0_rank8R1000/ \
     # --finetune /home/zhu.3723/kronvit/output/cifar100_vekron_bias0_rank4/best_checkpoint.pth \


end_time=$(date +"%s")

# Calculate and echo the total time taken
elapsed_time=$((end_time - start_time))
echo "Total execution time: $elapsed_time seconds"

# update A and B iteratively (1 time or 10 times)
# fix a1 a2 and b1 b2, find the relation between rank and accuracy
# how many epochs 
# using epoch \times flops
