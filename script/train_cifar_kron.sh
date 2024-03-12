#!/bin/bash

# SCRIPT_PATH =  "/home/zhu.3723/kronvit/"
cd /home/zhu.3723/kronvit/
GPU_NUM=1 
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=3


start_time=$(date +"%s") 

torchrun --nproc_per_node=$GPU_NUM \
     --master_port 29506\
      main.py \
     --epochs 300 \
     --model kron_deit_tiny_patch16_224 \
     --batch-size 512 \
     --data-set CIFAR \
     --data-path /local/storage/ding/cifar100 \
     --lr 1e-3\
     --finetune /home/zhu.3723/kronvit/output/cifar100_kron30/deit_tiny_patch16_224_6.0/best_checkpoint.pth \
     --output_dir /home/zhu.3723/kronvit/output/cifar100_kron30/deit_tiny_patch16_224_6.0/ \
     

end_time=$(date +"%s")

# Calculate and echo the total time taken
elapsed_time=$((end_time - start_time))
echo "Total execution time: $elapsed_time seconds"