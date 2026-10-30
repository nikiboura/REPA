#!/bin/bash

DATA_DIR="./data/chexpert_256_sdvae"
OUTPUT_DIR="./results"

# Same compute_metrics.py used by run/produce_counterfactuals/compute_metrics.sh --
# gives FID/KID/AUC/SSIM/PSNR/LPIPS directly comparable to the REPA/I2SB runs.
python scripts/counterfactual/compute_metrics.py \
  --data-dir $DATA_DIR \
  --generated-dir $OUTPUT_DIR/unsb-chexpert-baseline/counterfactuals \
  --split test
