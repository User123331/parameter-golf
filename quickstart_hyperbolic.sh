#!/usr/bin/env bash
# === Hyperbolic.ai Quick Start ===
# Run from ~/runpod-testing directory

set -euo pipefail
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
(while true; do sleep 60; nvidia-smi > /dev/null 2>&1; done) &
trap "kill $! 2>/dev/null" EXIT

GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
log "Detected ${GPU_COUNT} GPUs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Clone parameter-golf if needed
if [ ! -d "$HOME/parameter-golf" ]; then
    log "Cloning parameter-golf..."
    git clone https://github.com/openai/parameter-golf.git "$HOME/parameter-golf"
fi

# Install FA3 using pre-compiled .so
if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    log "Installing FA3 from pre-compiled .so..."

    # Clone flash-attention for Python interface
    if [ ! -d "$HOME/flash-attention" ]; then
        git clone https://github.com/Dao-AILab/flash-attention.git "$HOME/flash-attention"
    fi

    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

    # Copy pre-compiled .so
    mkdir -p "${SITE_PACKAGES}/flash_attn_3"
    cp "${SCRIPT_DIR}/compiled FA3/_C.abi3.so" "${SITE_PACKAGES}/flash_attn_3/"
    cp "${SCRIPT_DIR}/compiled FA3/flash_attn_config.py" "${SITE_PACKAGES}/flash_attn_3/"

    # Copy Python interface
    cp "$HOME/flash-attention/hopper/flash_attn_3/"*.py "${SITE_PACKAGES}/flash_attn_3/" 2>/dev/null || true

    # Install interface
    cd "$HOME/flash-attention/hopper"
    pip install -e . --no-build-isolation --break-system-packages 2>/dev/null || true

    # Symlink config to torch (fixes torch.compile backward crash)
    TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
    ln -sf "${SITE_PACKAGES}/flash_attn_3/flash_attn_config.py" "${TORCH_PATH}/flash_attn_config.py" 2>/dev/null || true

    python3 -c "from flash_attn_interface import flash_attn_func; print('FA3: OK')" || {
        log "WARNING: FA3 check failed"
    }
fi

# Download dataset
cd "$HOME/parameter-golf"
log "Downloading FineWeb dataset (8B tokens)..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# Symlink data to runpod-testing
cd "${SCRIPT_DIR}"
mkdir -p data/datasets data/tokenizers
[ ! -L "data/datasets/fineweb10B_sp1024" ] && \
    ln -s "$HOME/parameter-golf/data/datasets/fineweb10B_sp1024" data/datasets/
[ ! -L "data/tokenizers/fineweb_1024_bpe.model" ] && \
    ln -s "$HOME/parameter-golf/data/tokenizers/fineweb_1024_bpe.model" data/tokenizers/

log ""
log "=== Setup Complete ==="
log "GPUs: ${GPU_COUNT}"
log "FA3: $(python3 -c 'from flash_attn_interface import flash_attn_func; print("OK")' 2>/dev/null || echo 'FAILED')"
log "Dataset: $(ls -1 data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l) train shards"
log ""
log "Ready! Run:"
log "  MODE=mos bash run_mos_sota.sh"