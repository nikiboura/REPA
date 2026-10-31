#!/bin/bash

set -euo pipefail

UNSB_DATA_DIR="$(pwd)/data/chexpert_unsb"
CHECKPOINTS_DIR="$(pwd)/results/unsb-chexpert-baseline/checkpoints"
EXP_NAME="chexpert_h2pe_SB"

if [ "${1:-}" == "-s" ]; then
    N_EPOCHS=1
    N_EPOCHS_DECAY=0
else
    N_EPOCHS=400
    N_EPOCHS_DECAY=0
fi

# UNSB (Unpaired Neural Schrodinger Bridge), Healthy (A) -> Pleural Effusion (B),
# as an unpaired-translation baseline to compare against the I2SB/REPA counterfactual pipeline.
# --mode is deliberately omitted: it defaults to "sb" already, and UNSB's own
# options/sb_model.py has `choices='(FastCUT, fastcut, sb)'` -- a string, not a
# list, so argparse reads it as the individual characters '(', 'F', 'a', ... and
# rejects any value explicitly passed on the CLI. Defaults bypass choices
# validation, so relying on the default sidesteps their bug without patching it.
cd third_party/UNSB
python train.py \
  --dataroot $UNSB_DATA_DIR \
  --checkpoints_dir $CHECKPOINTS_DIR \
  --name $EXP_NAME \
  --direction AtoB \
  --lambda_SB 1.0 \
  --lambda_NCE 1.0 \
  --load_size 256 \
  --crop_size 256 \
  --n_epochs $N_EPOCHS \
  --n_epochs_decay $N_EPOCHS_DECAY \
  --gpu_ids 0
