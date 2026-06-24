#!/bin/bash

OUTPUT_DIR="./results"

if [ "$1" == "-1" ]; then
    MODEL="SiT-B/2"
    CKPT_STEP="4000000"
    NUM_SAMPLES=50000
elif [ "$1" == "-s" ]; then
    MODEL="SiT-S/4"
    CKPT_STEP="0000002"
    NUM_SAMPLES=4
else
    MODEL="SiT-S/4"
    CKPT_STEP="0400000"
    NUM_SAMPLES=50000
fi

# A — SD VAE
torchrun --nproc_per_node=1 generate.py \
  --model $MODEL \
  --ckpt $OUTPUT_DIR/sit-chexpert-sdvae/checkpoints/${CKPT_STEP}.pt \
  --sample-dir $OUTPUT_DIR/sit-chexpert-sdvae/samples \
  --num-classes 2 \
  --num-fid-samples $NUM_SAMPLES \
  --path-type linear \
  --mode ode \
  --num-steps 50 \
  --cfg-scale 1.5 \
  --resolution 256 \
  --vae mse

# B — MedVAE
torchrun --nproc_per_node=1 generate.py \
  --model $MODEL \
  --ckpt $OUTPUT_DIR/sit-chexpert-medvae/checkpoints/${CKPT_STEP}.pt \
  --sample-dir $OUTPUT_DIR/sit-chexpert-medvae/samples \
  --num-classes 2 \
  --num-fid-samples $NUM_SAMPLES \
  --path-type linear \
  --mode ode \
  --num-steps 50 \
  --cfg-scale 1.5 \
  --resolution 256 \
  --vae medvae

# C — MedVAE + REPA
torchrun --nproc_per_node=1 generate.py \
  --model $MODEL \
  --ckpt $OUTPUT_DIR/sit-chexpert-medvae-repa/checkpoints/${CKPT_STEP}.pt \
  --sample-dir $OUTPUT_DIR/sit-chexpert-medvae-repa/samples \
  --num-classes 2 \
  --num-fid-samples $NUM_SAMPLES \
  --path-type linear \
  --mode ode \
  --num-steps 50 \
  --cfg-scale 1.5 \
  --resolution 256 \
  --vae medvae \
  --projector-embed-dims 768

# D — SD VAE + REPA
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
