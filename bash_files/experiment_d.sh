#!/bin/bash

# D — SD VAE + REPA (MedDINOv3)

TEACHER_CKPT="/path/to/last.pt"
OUTPUT_DIR="./results"

if [ "$1" == "-1" ]; then
    MODEL="SiT-B/2"
    MAX_STEPS=4000000
    CKPT_STEPS=400000
    NUM_SAMPLES=50000
    CKPT_STEP="4000000"
elif [ "$1" == "-s" ]; then
    MODEL="SiT-S/4"
    MAX_STEPS=2
    CKPT_STEPS=2
    NUM_SAMPLES=4
    CKPT_STEP="0000002"
else
    MODEL="SiT-S/4"
    MAX_STEPS=400000
    CKPT_STEPS=400000
    NUM_SAMPLES=50000
    CKPT_STEP="0400000"
fi

echo "=== Train SD VAE + REPA ==="
python train.py \
  --exp-name sit-chexpert-sdvae-repa \
  --output-dir $OUTPUT_DIR \
  --model $MODEL \
  --num-classes 2 \
  --enc-type meddinov3-vit-b \
  --proj-coeff 0.5 \
  --vae-type sd \
  --data-dir ./data/chexpert_256_sdvae \
  --resolution 256 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --mixed-precision fp16 \
  --cfg-prob 0.1 \
  --path-type linear \
  --max-train-steps $MAX_STEPS \
  --checkpointing-steps $CKPT_STEPS \
  --sampling-steps 99999999 \
  --teacher-ckpt $TEACHER_CKPT \
  --report-to none

echo "=== Generate images==="
torchrun --nproc_per_node=1 generate.py \
  --model $MODEL \
  --ckpt $OUTPUT_DIR/sit-chexpert-sdvae-repa/checkpoints/${CKPT_STEP}.pt \
  --sample-dir $OUTPUT_DIR/sit-chexpert-sdvae-repa/samples \
  --num-classes 2 \
  --num-fid-samples $NUM_SAMPLES \
  --path-type linear \
  --mode ode \
  --num-steps 50 \
  --cfg-scale 1.5 \
  --resolution 256 \
  --projector-embed-dims 768 \
  --vae mse
