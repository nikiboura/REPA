#!/bin/bash

set -euo pipefail

DATA_DIR="./data/chexpert_256_sdvae"
OUTPUT_DIR="./results"

# FID, KID, SSIM, PSNR, LPIPS, and AUC -- AUC needs VAE latents, which only
# chexpert_256_sdvae has (built by run/produce_counterfactuals/prepare_data.sh)
# Run that first if this errors below.
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: $DATA_DIR not found. Run 'bash run/produce_counterfactuals/prepare_data.sh' first -- AUC needs its VAE latents." >&2
    exit 1
fi

# UNSB was trained/tested on chexpert_256_unsb but is scored against
# chexpert_256_sdvae. Both come from the same seeded split in
# prepare_chexpert.py, so their test sets must be byte-identical -- otherwise
# the reference set differs from (or overlaps with) what UNSB saw.
UNSB_DATA_DIR="./data/chexpert_256_unsb"
if [ -d "$UNSB_DATA_DIR" ]; then
    if ! cmp -s "$UNSB_DATA_DIR/vae-sd/test/dataset.json" "$DATA_DIR/vae-sd/test/dataset.json" \
        || ! diff -rq "$UNSB_DATA_DIR/images/test" "$DATA_DIR/images/test" > /dev/null; then
        echo "ERROR: test split of $UNSB_DATA_DIR differs from $DATA_DIR -- rebuild both from the same CheXpert copy." >&2
        exit 1
    fi
fi

python scripts/counterfactual/compute_metrics.py \
  --data-dir $DATA_DIR \
  --generated-dir $OUTPUT_DIR/unsb-chexpert-baseline/counterfactuals \
  --split test
