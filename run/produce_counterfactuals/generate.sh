#!/bin/bash

OUTPUT_DIR="./results"
DATA_DIR="./data/chexpert_256_sdvae"

if [ "$1" == "-s" ]; then
    MODEL="SiT-S/4"
    CKPT_STEP="0000002"
    NUM_SAMPLES=4
else
    MODEL="SiT-S/4"
    CKPT_STEP="0400000"
    NUM_SAMPLES=500
fi

# REPA & no cond-x1
python scripts/counterfactual/generate_counterfactual.py \
  --ckpt $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-repa/checkpoints/${CKPT_STEP}.pt \
  --data-dir $DATA_DIR \
  --output-dir $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-repa/counterfactuals \
  --model $MODEL \
  --num-classes 2 \
  --split test \
  --vae mse \
  --projector-embed-dims 768 \
  --num-steps 200 \
  --beta-max 0.3 \
  --cfg-scale 1.5 \
  --num-samples $NUM_SAMPLES \
  --batch-size 2

# REPA & condx1
python scripts/counterfactual/generate_counterfactual.py \
  --ckpt $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-repa-condx1/checkpoints/${CKPT_STEP}.pt \
  --data-dir $DATA_DIR \
  --output-dir $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-repa-condx1/counterfactuals \
  --model $MODEL \
  --num-classes 2 \
  --split test \
  --vae mse \
  --projector-embed-dims 768 \
  --num-steps 200 \
  --beta-max 0.3 \
  --cond-x1 \
  --cfg-scale 1.5 \
  --num-samples $NUM_SAMPLES \
  --batch-size 2

# no REPA & no cond-x1
python scripts/counterfactual/generate_counterfactual.py \
  --ckpt $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-norepa/checkpoints/${CKPT_STEP}.pt \
  --data-dir $DATA_DIR \
  --output-dir $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-norepa/counterfactuals \
  --model $MODEL \
  --num-classes 2 \
  --split test \
  --vae mse \
  --num-steps 200 \
  --beta-max 0.3 \
  --cfg-scale 1.5 \
  --num-samples $NUM_SAMPLES \
  --batch-size 2

# no REPA & cond-x1
python scripts/counterfactual/generate_counterfactual.py \
  --ckpt $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-norepa-condx1/checkpoints/${CKPT_STEP}.pt \
  --data-dir $DATA_DIR \
  --output-dir $OUTPUT_DIR/i2sb-sit-s4-chexpert-sdvae-norepa-condx1/counterfactuals \
  --model $MODEL \
  --num-classes 2 \
  --split test \
  --vae mse \
  --num-steps 200 \
  --beta-max 0.3 \
  --cond-x1 \
  --cfg-scale 1.5 \
  --num-samples $NUM_SAMPLES \
  --batch-size 2
