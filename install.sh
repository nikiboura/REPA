#!/bin/bash

set -euo pipefail

# Single install script for every experiment pipeline in this repo
# (model_ablations/, produce_counterfactuals/, unsb_baseline/).

conda create -n repa python=3.9 -y
conda activate repa

pip install -r requirements.txt
pip install medvae --no-deps

# unsb_baseline/ only: pulls the UNSB submodule + its extra deps. UNSB's own
# README pins an old torch/cu111 build, but the codebase has no hard
# dependency on it -- the REPA env's torch above works fine.
git submodule update --init --recursive third_party/UNSB
pip install dominate visdom gputil packaging
