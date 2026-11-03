#!/bin/bash

set -euo pipefail

# Single install script for every experiment pipeline in this repo
# (model_ablations/, produce_counterfactuals/, unsb_baseline/).

conda create -n repa python=3.9 -y

# `conda activate` needs conda's shell functions loaded into the current
# shell -- normally wired up by `conda init` into ~/.bash_profile, but that
# file is only sourced by an interactive login shell, not by `bash install.sh`
# running as a plain script. Source conda's hook directly instead, so this
# works regardless of whether `conda init` has ever been run on this machine.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate repa

pip install -r requirements.txt
pip install medvae --no-deps

# unsb_baseline/ only: pulls the UNSB submodule + its extra deps. UNSB's own
# README pins an old torch/cu111 build, but the codebase has no hard
# dependency on it -- the REPA env's torch above works fine.
git submodule update --init --recursive third_party/UNSB
pip install dominate visdom gputil packaging
