#!/bin/bash

# SimpleBEV PyTorch Lightning Training Script
# Usage: bash run_train.sh

echo "=== SimpleBEV Training with PyTorch Lightning ==="
echo "Starting training at $(date)"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/zc/simple_bev:/home/zc/vggt:$PYTHONPATH

# Config file
CONFIG_PATH="./configs/simplebev_seg.yaml"

echo "Config: $CONFIG_PATH"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"

# Run training
python train_nuscenes_lightning.py

echo "Training completed at $(date)"
