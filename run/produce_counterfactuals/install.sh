#!/bin/bash
conda create -n repa python=3.9 -y
conda activate repa
pip install -r requirements.txt
pip install medvae --no-deps
