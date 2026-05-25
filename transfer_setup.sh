#!/bin/bash
# ATR-UMOD Environment Transfer Setup
# Run on target PC after copying the project directory.
# Usage: bash transfer_setup.sh

set -e

echo "=== ATR-UMOD Environment Transfer ==="

# --- Option A: Unpack conda-pack archive (fast, exact copy) ---
if [ -f "atrumod_env.tar.gz" ]; then
    echo "Found conda-pack archive. Extracting..."
    mkdir -p "$HOME/miniconda3/envs/atrumod" 2>/dev/null || \
    mkdir -p "$HOME/anaconda3/envs/atrumod" 2>/dev/null || true
    tar -xzf atrumod_env.tar.gz -C "${CONDA_PREFIX:-$HOME/miniconda3/envs/atrumod}"
    echo "Environment extracted. Activate with: conda activate $CONDA_PREFIX/../envs/atrumod"
    exit 0
fi

# --- Option B: Rebuild from environment.yml (smaller transfer) ---
echo "No conda-pack archive found. Rebuilding from environment.yml..."

# 1. Create Python 3.10 env
conda create -n atrumod python=3.10 -y
source activate atrumod 2>/dev/null || conda activate atrumod

# 2. Install PyTorch — target: CUDA 12.1 or 12.8 depending on GPU
#    For RTX 50-series (Blackwell), use CUDA 12.8:
CUDA_VER="${CUDA_VERSION:-cu128}"
if [ "$CUDA_VER" = "cu128" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# 3. Install mmcv from prebuilt wheels (must match CUDA + torch version)
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/${CUDA_VER}/torch2.7/index.html

# 4. Install remaining packages
pip install mmengine mmdet mmrotate
pip install tensorboard einops opencv-python matplotlib tqdm pycocotools

# 5. Patch mmrotate version check (accepts newer mmcv)
MMROTATE_INIT=$(python -c "import mmrotate; print(mmrotate.__file__)" | sed 's/__pycache__.*/__init__.py/')
sed -i "s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/" "$MMROTATE_INIT"
sed -i "s/mmdet_maximum_version = '3.1.0'/mmdet_maximum_version = '3.5.0'/" "$MMROTATE_INIT"

# 6. Install project
pip install -e .

echo ""
echo "=== Done. Activate with: conda activate atrumod ==="
echo "=== Verify with: python tools/check_env.py ==="
