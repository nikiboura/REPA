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

    # Resume from the highest numbered <N>_net_G.pth snapshot. UNSB writes these
    # at the end of every 5th epoch (--save_epoch_freq), together with
    # <N>_net_{F,D,E}.pth, so N is known to be a fully completed epoch. We load
    # that snapshot explicitly (--epoch N) rather than latest_net_*.pth, which
    # is also overwritten mid-epoch every 5000 iters and so belongs to an
    # unknown, partially trained epoch -- pairing it with an epoch count read
    # from loss_log.txt would silently skip epochs and advance the LR schedule.
    # At most 4 epochs of progress are redone. (UNSB doesn't save optimizer
    # state, so Adam restarts on resume either way -- upstream behavior.)
    SNAPSHOT_EPOCH=$(find "$EXP_DIR" -maxdepth 1 -name '[0-9]*_net_G.pth' -exec basename {} \; 2>/dev/null \
        | sed 's/_net_G\.pth//' | sort -n | tail -1) || true
    SNAPSHOT_EPOCH=${SNAPSHOT_EPOCH:-0}
    if [ "$SNAPSHOT_EPOCH" -gt 0 ]; then
        for NET in F D E; do
            if [ ! -f "$EXP_DIR/${SNAPSHOT_EPOCH}_net_${NET}.pth" ]; then
                echo "ERROR: $EXP_DIR/${SNAPSHOT_EPOCH}_net_${NET}.pth missing -- incomplete snapshot." >&2
                exit 1
            fi
        done
        if [ "$SNAPSHOT_EPOCH" -ge "$((N_EPOCHS + N_EPOCHS_DECAY))" ]; then
            echo "Snapshot already covers epoch $SNAPSHOT_EPOCH >= target $((N_EPOCHS + N_EPOCHS_DECAY)) -- nothing to resume."
            exit 0
        fi
        EPOCH_COUNT=$((SNAPSHOT_EPOCH + 1))
        echo "Found snapshot of completed epoch $SNAPSHOT_EPOCH -- resuming from epoch $EPOCH_COUNT"
        RESUME_FLAGS="--continue_train --epoch $SNAPSHOT_EPOCH --epoch_count $EPOCH_COUNT"
    elif [ -f "$EXP_DIR/latest_net_G.pth" ]; then
        echo "Only latest_net_*.pth found (stopped before epoch 5) -- its epoch is unknown, training from scratch."
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
