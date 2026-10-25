#!/bin/bash

TEACHER_CKPT="/path/to/last.pt"
OUTPUT_DIR="./results"

if [ "$1" == "-1" ]; then
    MODEL="SiT-B/2"
    MAX_STEPS=400000
    CKPT_STEPS=400000
elif [ "$1" == "-s" ]; then
    MODEL="SiT-S/4"
    MAX_STEPS=2
    CKPT_STEPS=2
else
    MODEL="SiT-S/4"
    MAX_STEPS=400000
    CKPT_STEPS=400000
fi

# A — REPA
python scripts/counterfactual/train.py \
  --exp-name i2sb-sit-s4-chexpert-sdvae-repa \
  --output-dir $OUTPUT_DIR \
  --model $MODEL \
  --num-classes 2 \
  --enc-type meddinov3-vit-b \
  --proj-coeff 0.5 \
  --vae-type sd \
  --data-dir ./data/chexpert_256_sdvae \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --mixed-precision fp16 \
  --path-type linear \
  --beta-max 0.3 \
  --max-train-steps $MAX_STEPS \
  --checkpointing-steps $CKPT_STEPS \
  --sampling-steps 99999999 \
  --teacher-ckpt $TEACHER_CKPT \
  --report-to none


# B — REPA & condx1
python scripts/counterfactual/train.py \
  --exp-name i2sb-sit-s4-chexpert-sdvae-repa-condx1 \
  --output-dir $OUTPUT_DIR \
  --model $MODEL \
  --num-classes 2 \
  --enc-type meddinov3-vit-b \
  --proj-coeff 0.5 \
  --vae-type sd \
  --data-dir ./data/chexpert_256_sdvae \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --mixed-precision fp16 \
  --path-type linear \
  --beta-max 0.3 \
  --cond-x1 \
  --max-train-steps $MAX_STEPS \
  --checkpointing-steps $CKPT_STEPS \
  --sampling-steps 99999999 \
  --teacher-ckpt $TEACHER_CKPT \
  --report-to none

# C — No REPA 
python scripts/counterfactual/train.py \
  --exp-name i2sb-sit-s4-chexpert-sdvae-norepa \
  --output-dir $OUTPUT_DIR \
  --model $MODEL \
  --num-classes 2 \
  --enc-type None \
  --proj-coeff 0.0 \
  --vae-type sd \
  --data-dir ./data/chexpert_256_sdvae \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --mixed-precision fp16 \
  --path-type linear \
  --beta-max 0.3 \
  --max-train-steps $MAX_STEPS \
  --checkpointing-steps $CKPT_STEPS \
  --sampling-steps 99999999 \
  --teacher-ckpt $TEACHER_CKPT \
  --report-to none
  

# D — No REPA & condx1
python scripts/counterfactual/train.py \
  --exp-name i2sb-sit-s4-chexpert-sdvae-norepa-condx1 \
  --output-dir $OUTPUT_DIR \
  --model $MODEL \
  --num-classes 2 \
  --enc-type None \
  --proj-coeff 0.0 \
  --vae-type sd \
  --data-dir ./data/chexpert_256_sdvae \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --mixed-precision fp16 \
  --path-type linear \
  --beta-max 0.3 \
  --cond-x1 \
  --max-train-steps $MAX_STEPS \
  --checkpointing-steps $CKPT_STEPS \
  --sampling-steps 99999999 \
  --teacher-ckpt $TEACHER_CKPT \
  --report-to none