#!/bin/bash

set -euo pipefail

DATA_DIR="./data/chexpert_256_sdvae"
OUTPUT_DIR="./results"

# FID, KID, SSIM, PSNR, LPIPS, and AUC -- AUC needs VAE latents, which only
# chexpert_256_sdvae has (built by run/produce_counterfactuals/prepare_data.sh,
# a separate, heavier pipeline). Run that first if this errors below.
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: $DATA_DIR not found. Run 'bash run/produce_counterfactuals/prepare_data.sh' first -- AUC needs its VAE latents." >&2
    exit 1
fi

python scripts/counterfactual/compute_metrics.py \
  --data-dir $DATA_DIR \
  --generated-dir $OUTPUT_DIR/unsb-chexpert-baseline/counterfactuals \
  --split test
