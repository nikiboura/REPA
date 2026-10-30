#!/bin/bash

KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
CHEXPERT_ROOT="./data/chexpert"
DATA_DIR="./data/chexpert_256_unsb"
UNSB_DATA_DIR="./data/chexpert_unsb"

# UNSB trains in pixel space and never reads VAE latents, so --skip-vae builds
# a lean Healthy/Pleural-Effusion split (resized PNGs + labels only) without
# downloading or running the SD-VAE -- much faster than
# run/produce_counterfactuals/prepare_data.sh, which also builds the latents
# needed elsewhere. Skips re-downloading/re-resizing if already built.
if [ ! -d "$DATA_DIR/images" ]; then
  python scripts/counterfactual/prepare_chexpert.py \
    --chexpert-root $CHEXPERT_ROOT \
    --out-dir $DATA_DIR \
    --skip-vae \
    --resolution 256 \
    --kaggle-json $KAGGLE_JSON \
    --pathologies "Pleural Effusion" \
    --test-size 500
fi

# Re-sort into UNSB's trainA/trainB/testA/testB layout.
python scripts/counterfactual/build_unsb_dataset.py \
  --data-dir $DATA_DIR \
  --out-dir $UNSB_DATA_DIR \
  --label-a 0 \
  --label-b 1
