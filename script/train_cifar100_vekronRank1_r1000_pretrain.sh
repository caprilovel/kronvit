#!/bin/bash

# SCRIPT_PATH =  "/home/zhu.3723/kronvit/"
cd /home/zhu.3723/kronvit/
GPU_NUM=2
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1



start_time=$(date +"%s") 

torchrun --nproc_per_node=$GPU_NUM \
     --master_port 29501\
      main.py \
     --epochs 1500 \
     --model vekron_deit_tiny_patch16_224 \
     --batch-size 512 \
     --k1l \
     --data-set CIFAR \
     --data-path /local/storage/ding/cifar100 \
     --lr 1e-4\
     --warmup-lr 5e-5\
     --min-lr 5e-5\
     --output_dir /local/storage/ding/kronvit/tiny \
     --finetune /local/storage/ding/kronvit/tiny/best_checkpoint.pth \



end_time=$(date +"%s")

# Calculate and echo the total time taken
elapsed_time=$((end_time - start_time))
echo "Total execution time: $elapsed_time seconds"

# update A and B iteratively (1 time or 10 times)
# fix a1 a2 and b1 b2, find the relation between rank and accuracy
# how many epochs 
# using epoch \times flops
