#!/bin/bash

# SCRIPT_PATH =  "/home/zhu.3723/kronvit/"
cd /home/zhu.3723/kronvit/
GPU_NUM=2
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6,7



start_time=$(date +"%s") 

torchrun --nproc_per_node=$GPU_NUM \
     --master_port 29503\
      main.py \
     --epochs 500 \
     --model k1l_deit_tiny_patch16_224 \
     --batch-size 512 \
     --k1l \
     --data-set CIFAR \
     --data-path /local/storage/ding/cifar100 \
     --lr 3e-4\
     --warmup-lr 1e-4\
     --min-lr 1e-4\
     --output_dir /home/zhu.3723/kronvit/output/cifar100_k1l_bias0/deit_tiny_patch16_224_1 \
     --kron_rank 1\
     --non-sparse \
     --finetune /home/zhu.3723/kronvit/output/cifar100_k1l_bias0/deit_tiny_patch16_224_1/best_checkpoint.pth \



end_time=$(date +"%s")

# Calculate and echo the total time taken
elapsed_time=$((end_time - start_time))
echo "Total execution time: $elapsed_time seconds"

# copy this script to the output file folder
cp /home/zhu.3723/kronvit/script/train_cifar100_k1l_2.sh /home/zhu.3723/kronvit/output/cifar100_k1l_bias0/deit_tiny_patch16_224_1/

# update A and B iteratively (1 time or 10 times)
# fix a1 a2 and b1 b2, find the relation between rank and accuracy
# how many epochs 
# using epoch \times flops
