#!/bin/bash

# SCRIPT_PATH =  "/home/zhu.3723/kronvit/"
cd /home/zhu.3723/kronvit/
GPU_NUM=2
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3


start_time=$(date +"%s") 

torchrun --nproc_per_node=$GPU_NUM \
     --master_port 29511\
      main.py \
     --epochs 30 \
     --model kron_deit_tiny_patch16_224 \
     --batch-size 128 \
     --kron \
     --kron_b_freeze \
     --data-set CIFAR \
     --data-path /local/storage/ding/cifar100 \
     --lr 1e-3\
     --warmup-lr 1e-4\
     --min-lr 1e-4\
     --output_dir /home/zhu.3723/kronvit/output/cifar100_kron30/deit_tiny_patch16_224_40/ \
     --kron_rank 40\
     --shape_bias 3\
    --finetune /home/zhu.3723/kronvit/output/cifar100_kron30/deit_tiny_patch16_224_40/best_checkpoint.pth \

torchrun --nproc_per_node=$GPU_NUM \
     --master_port 29512\
      main.py \
     --epochs 30 \
     --model kron_deit_tiny_patch16_224 \
     --batch-size 128 \
     --kron \
     --kron_a_freeze \
     --data-set CIFAR \
     --data-path /local/storage/ding/cifar100 \
     --lr 1e-3\
     --warmup-lr 1e-4\
     --min-lr 1e-4\
     --output_dir /home/zhu.3723/kronvit/output/cifar100_kron30/deit_tiny_patch16_224_40/ \
     --kron_rank 40\
     --shape_bias 3\
     --finetune /home/zhu.3723/kronvit/output/cifar100_kron30/deit_tiny_patch16_224_40/best_checkpoint.pth \

end_time=$(date +"%s")

# Calculate and echo the total time taken
elapsed_time=$((end_time - start_time))
echo "Total execution time: $elapsed_time seconds"