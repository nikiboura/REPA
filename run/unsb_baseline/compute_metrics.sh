#!/bin/bash

DATA_DIR="./data/chexpert_256_unsb"
OUTPUT_DIR="./results"

# Same compute_metrics.py used by run/produce_counterfactuals/compute_metrics.sh,
# but with --skip-latents: our chexpert_256_unsb data has no VAE latents (see
# prepare_data.sh), and FID/KID/SSIM/PSNR/LPIPS don't need them -- only the AUC
# metric does, so that's the one number this skips.
#
# For the AUC too, point --data-dir at an existing chexpert_256_sdvae (built by
# run/produce_counterfactuals/prepare_data.sh) and drop --skip-latents instead.
python scripts/counterfactual/compute_metrics.py \
  --data-dir $DATA_DIR \
  --generated-dir $OUTPUT_DIR/unsb-chexpert-baseline/counterfactuals \
  --split test \
  --skip-latents
