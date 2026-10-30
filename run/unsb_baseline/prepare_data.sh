#!/bin/bash

DATA_DIR="./data/chexpert_256_sdvae"
UNSB_DATA_DIR="./data/chexpert_unsb"

# Reuses the images already prepared by run/produce_counterfactuals/prepare_data.sh --
# UNSB only needs the PNGs (it trains in pixel space, no VAE latents involved.
# Just re-sorts label 0 (Healthy) / label 1 (Pleural Effusion) into trainA/trainB/testA/testB.
python scripts/counterfactual/build_unsb_dataset.py \
  --data-dir $DATA_DIR \
  --out-dir $UNSB_DATA_DIR \
  --label-a 0 \
  --label-b 1
