#!/bin/bash

UNSB_DATA_DIR="$(pwd)/data/chexpert_unsb"
CHECKPOINTS_DIR="$(pwd)/results/unsb-chexpert-baseline/checkpoints"
UNSB_RESULTS_DIR="$(pwd)/results/unsb-chexpert-baseline/unsb_raw"
OUTPUT_DIR="$(pwd)/results/unsb-chexpert-baseline/counterfactuals"
EXP_NAME="chexpert_h2pe_SB"

if [ "$1" == "-s" ]; then
    NUM_TEST=4
else
    NUM_TEST=500
fi

cd third_party/UNSB
# --mode omitted: see the comment in train.sh -- defaults to "sb" already,
# UNSB's own choices= for --mode is a buggy string (not a list), so passing
# it explicitly on the CLI gets rejected.
python test.py \
  --dataroot $UNSB_DATA_DIR \
  --checkpoints_dir $CHECKPOINTS_DIR \
  --results_dir $UNSB_RESULTS_DIR \
  --name $EXP_NAME \
  --direction AtoB \
  --phase test \
  --epoch latest \
  --eval \
  --num_test $NUM_TEST \
  --gpu_ids 0
cd - > /dev/null

# UNSB writes fake_1 .. fake_5 (one per NFE / refinement step). We use the
# final step (fake_5, NFE=5) as "the" translation, matching the paper's
# reported NFE=5 best setting, and reshape it to match compute_metrics.py's
# <idx>_healthy.png / <idx>_pe.png pairing convention.
python scripts/counterfactual/reformat_unsb_output.py \
  --unsb-results-dir $UNSB_RESULTS_DIR/$EXP_NAME/test_latest/images \
  --out-dir $OUTPUT_DIR \
  --nfe 5
