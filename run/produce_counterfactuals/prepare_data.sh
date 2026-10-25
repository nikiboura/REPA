#!/bin/bash
KAGGLE_JSON="/path/to/kaggle.json"

# Preparation script for counterfactual generation
# Only pleuran effusion / healthy patients are downloaded
# 
python scripts/counterfactual/prepare_chexpert.py \
  --chexpert-root ./data/chexpert \
  --out-dir ./data/chexpert_256_sdvae \
  --vae-type sd \
  --resolution 256 \
  --kaggle-json $KAGGLE_JSON \
  --pathologies "Pleural Effusion" \
  --test-size 500



