#!/bin/bash
# ATR-UMOD Environment Setup
# Run: bash setup_env.sh

set -e

ENV_NAME="atrumod"
PYTHON_VERSION="3.10"

echo "=== Creating conda environment ${ENV_NAME} (Python ${PYTHON_VERSION}) ==="
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

echo "=== Installing PyTorch with CUDA 12.1 ==="
conda run -n ${ENV_NAME} pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing MMRotate stack ==="
conda run -n ${ENV_NAME} pip install mmengine

# mmcv with prebuilt wheel for cu121/torch2.x
conda run -n ${ENV_NAME} pip install mmcv==2.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

conda run -n ${ENV_NAME} pip install mmdet mmrotate

echo "=== Installing project dependencies ==="
conda run -n ${ENV_NAME} pip install tensorboard opencv-python matplotlib tqdm pycocotools

echo ""
echo "=== Done! Activate with: conda activate ${ENV_NAME} ==="
echo "=== Verify: python -c 'import mmrotate; print(mmrotate.__version__)' ==="
