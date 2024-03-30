cd /home/zhu.3723/kronvit/
GPU_NUM=2
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3 


start_time=$(date +"%s")
torchrun --nproc_per_node=$GPU_NUM \
     --master_port 29521\
     main.py  \
     --model deit_tiny_patch16_224 \
     --batch-size 512 \
     --data-set CIFAR \
     --data-path /local/storage/ding/cifar100 \
     --output_dir /home/zhu.3723/kronvit/output/cifar100_train_common30/ \
     # --finetune /home/zhu.3723/kronvit/output/cifar100_train_common30/best_checkpoint.pth \

end_time=$(date +"%s")

# Calculate and echo the total time taken
elapsed_time=$((end_time - start_time))
echo "Total execution time: $elapsed_time seconds"