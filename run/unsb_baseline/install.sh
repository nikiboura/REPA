#!/bin/bash

# Pulls the UNSB submodule and installs its extra deps into the existing
# `repa` conda env. UNSB's own README pins an old torch/cu111 build, but
# the codebase has no hard dependency on it -- the REPA env's torch works fine.

git submodule update --init --recursive third_party/UNSB

conda activate repa
pip install dominate visdom gputil packaging
