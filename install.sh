#!/bin/bash

set -euo pipefail

# Point everything at a scratch disk: export SCRATCH_DIR=/path/to/scratch && bash install.sh
# (also clone and run this repo itself from inside $SCRATCH_DIR, since every
# script here saves data/checkpoints to repo-relative ./data and ./results)

if [ -n "${SCRATCH_DIR:-}" ]; then
    mkdir -p "$SCRATCH_DIR/conda-envs" "$SCRATCH_DIR/caches/pip" "$SCRATCH_DIR/caches/huggingface" "$SCRATCH_DIR/caches/torch"
    export PIP_CACHE_DIR="$SCRATCH_DIR/caches/pip" HF_HOME="$SCRATCH_DIR/caches/huggingface" TORCH_HOME="$SCRATCH_DIR/caches/torch"
    CONDA_ENV="$SCRATCH_DIR/conda-envs/repa"
    conda create --prefix "$CONDA_ENV" python=3.9 -y
else
    CONDA_ENV="repa"
    conda create -n "$CONDA_ENV" python=3.9 -y
fi

# Lets `conda activate` work even if `conda init` was never run in this shell.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install -r requirements.txt
pip install medvae --no-deps

# For unsb_baseline/ only.
git submodule update --init --recursive third_party/UNSB
pip install dominate visdom gputil packaging

echo "Done. Next time, activate with: conda activate $CONDA_ENV"
