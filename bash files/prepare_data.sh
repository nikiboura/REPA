#!/bin/bash

MAX_SAMPLES=80000
KAGGLE_JSON="/path/to/kaggle.json"

if [ "$1" == "-1" ]; then
    MAX_SAMPLES_ARG=""
elif [ "$1" == "-s" ]; then
    MAX_SAMPLES_ARG="--max-samples 10"
else
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi

# Experiment A (SD VAE)
python dataset_preparation_scripts/prepare_chexpert.py \
  --chexpert-root ./data/chexpert \
  --out-dir ./data/chexpert_256_sdvae \
  --vae-type sd \
  --resolution 256 \
  --kaggle-json $KAGGLE_JSON \
  $MAX_SAMPLES_ARG

# Experiments B & C (MedVAE)
python dataset_preparation_scripts/prepare_chexpert.py \
  --chexpert-root ./data/chexpert \
  --out-dir ./data/chexpert_256_medvae \
  --vae-type medvae \
  --resolution 256 \
  --kaggle-json $KAGGLE_JSON \
  $MAX_SAMPLES_ARG
