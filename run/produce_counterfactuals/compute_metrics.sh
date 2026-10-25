#!/bin/bash

OUTPUT_DIR="./results"
DATA_DIR="./data/chexpert_256_sdvae"

# REPA & no cond-x1
python scripts/counterfactual/compute_metrics.py \
  --data-dir $DATA_DIR \
  --generated-dir $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-repa/counterfactuals \
  --split test 


# REPA & condx1
python scripts/counterfactual/compute_metrics.py \
  --data-dir $DATA_DIR \
  --generated-dir $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-repa-condx1/counterfactuals \
  --split test 


# no REPA & no cond-x1
python scripts/counterfactual/compute_metrics.py \
  --data-dir $DATA_DIR \
  --generated-dir $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-norepa/counterfactuals \
  --split test 


# no REPA & cond-x1
python scripts/counterfactual/compute_metrics.py \
  --data-dir $DATA_DIR \
  --generated-dir $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-norepa-condx1/counterfactuals \
  --split test 
