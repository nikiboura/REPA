#!/bin/bash

set -euo pipefail

pip install -q torch-fidelity

RESULTS_DIR="./results"
DATA_DIR="./data"

SAMPLES_A=$(ls -d $RESULTS_DIR/sit-chexpert-sdvae/samples/*/ 2>/dev/null | sort | tail -1)
SAMPLES_B=$(ls -d $RESULTS_DIR/sit-chexpert-medvae/samples/*/ 2>/dev/null | sort | tail -1)
SAMPLES_C=$(ls -d $RESULTS_DIR/sit-chexpert-medvae-repa/samples/*/ 2>/dev/null | sort | tail -1)
SAMPLES_D=$(ls -d $RESULTS_DIR/sit-chexpert-sdvae-repa/samples/*/ 2>/dev/null | sort | tail -1)

REF=$DATA_DIR/chexpert_256_sdvae/images/train

python scripts/compute_metrics.py \
  --ref $REF \
  --samples "SD VAE=$SAMPLES_A" "MedVAE=$SAMPLES_B" "MedVAE + REPA=$SAMPLES_C" "SD VAE + REPA=$SAMPLES_D"
