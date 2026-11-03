#!/bin/bash

set -euo pipefail

UNSB_DATA_DIR="$(pwd)/data/chexpert_unsb"
CHECKPOINTS_DIR="$(pwd)/results/unsb-chexpert-baseline/checkpoints"
EXP_NAME="chexpert_h2pe_SB"

EXP_DIR="$CHECKPOINTS_DIR/$EXP_NAME"
RESUME_FLAGS=""

if [ "${1:-}" == "-s" ]; then
    N_EPOCHS=1
    N_EPOCHS_DECAY=0
else
    # 200 + 200 = 400 total epochs, matching the paper: constant LR for the
    # first 200, then linearly decayed to zero over the last 200 (this is
    # also UNSB's own code default).
    N_EPOCHS=200
    N_EPOCHS_DECAY=200 

    # Resume from an earlier checkpoint if one exists. Epoch count comes from
    # loss_log.txt's "(epoch: N, ...)" lines, floored by the highest numbered
    # <N>_net_G.pth snapshot in case the log is missing.
    if [ -f "$EXP_DIR/latest_net_G.pth" ]; then
        LOG_EPOCH=$(grep -oE '^\(epoch: [0-9]+' "$EXP_DIR/loss_log.txt" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1) || true
        LOG_EPOCH=${LOG_EPOCH:-0}
        SNAPSHOT_EPOCH=$(find "$EXP_DIR" -maxdepth 1 -name '[0-9]*_net_G.pth' -exec basename {} \; 2>/dev/null \
            | sed 's/_net_G\.pth//' | sort -n | tail -1) || true
        SNAPSHOT_EPOCH=${SNAPSHOT_EPOCH:-0}
        LAST_EPOCH=$LOG_EPOCH
        if [ "$SNAPSHOT_EPOCH" -gt "$LAST_EPOCH" ]; then
            LAST_EPOCH=$SNAPSHOT_EPOCH
        fi
        EPOCH_COUNT=$((LAST_EPOCH + 1))
        if [ "$EPOCH_COUNT" -gt "$((N_EPOCHS + N_EPOCHS_DECAY))" ]; then
            echo "Checkpoint already covers epoch $LAST_EPOCH >= target $((N_EPOCHS + N_EPOCHS_DECAY)) -- nothing to resume."
            exit 0
        fi
        echo "Found existing checkpoint (last completed epoch: $LAST_EPOCH) -- resuming from epoch $EPOCH_COUNT"
        RESUME_FLAGS="--continue_train --epoch_count $EPOCH_COUNT"
    fi
fi

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
  --gpu_ids 0 \
  --display_id 0 \
  $RESUME_FLAGS
