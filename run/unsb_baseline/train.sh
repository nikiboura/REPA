#!/bin/bash

set -euo pipefail

UNSB_DATA_DIR="$(pwd)/data/chexpert_unsb"
CHECKPOINTS_DIR="$(pwd)/results/unsb-chexpert-baseline/checkpoints"
EXP_NAME="chexpert_h2pe_SB"

EXP_DIR="$CHECKPOINTS_DIR/$EXP_NAME"
RESUME_FLAGS=()

if [ "${1:-}" == "-s" ]; then
    N_EPOCHS=1
    N_EPOCHS_DECAY=0
else
    N_EPOCHS=400
    N_EPOCHS_DECAY=0

    # Full run: if a checkpoint from an earlier (interrupted) session is sitting
    # on Drive, resume from it instead of restarting. UNSB always keeps
    # latest_net_*.pth current (every save_epoch_freq epochs, and mid-epoch via
    # save_latest_freq), so --continue_train (loads --epoch latest, the
    # default) is always safe; --epoch_count just tells its training loop
    # where to resume counting, read from the last "End of epoch N" line in
    # its own loss_log.txt.
    if [ -f "$EXP_DIR/latest_net_G.pth" ]; then
        LAST_EPOCH=$(grep -o '^End of epoch [0-9]*' "$EXP_DIR/loss_log.txt" 2>/dev/null | awk '{print $4}' | sort -n | tail -1) || true
        LAST_EPOCH=${LAST_EPOCH:-0}
        EPOCH_COUNT=$((LAST_EPOCH + 1))
        if [ "$EPOCH_COUNT" -gt "$((N_EPOCHS + N_EPOCHS_DECAY))" ]; then
            echo "Checkpoint already covers epoch $LAST_EPOCH >= target $((N_EPOCHS + N_EPOCHS_DECAY)) -- nothing to resume."
            exit 0
        fi
        echo "Found existing checkpoint (last completed epoch: $LAST_EPOCH) -- resuming from epoch $EPOCH_COUNT"
        RESUME_FLAGS=(--continue_train --epoch_count "$EPOCH_COUNT")
    fi
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
  --gpu_ids 0 \
  "${RESUME_FLAGS[@]}"
