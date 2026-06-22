#!/bin/bash

pip install -q torch-fidelity

RESULTS_DIR="./results"
DATA_DIR="./data"

SAMPLES_A=$(ls -d $RESULTS_DIR/sit-chexpert-sdvae/samples/*/ 2>/dev/null | sort | tail -1)
SAMPLES_B=$(ls -d $RESULTS_DIR/sit-chexpert-medvae/samples/*/ 2>/dev/null | sort | tail -1)
SAMPLES_C=$(ls -d $RESULTS_DIR/sit-chexpert-medvae-repa/samples/*/ 2>/dev/null | sort | tail -1)

REF=$DATA_DIR/chexpert_256_sdvae/images/train

echo "=== FID (torch-fidelity) ==="

echo "[A] SD VAE"
fidelity --gpu 0 --fid --samples-find-deep \
  --input1 "$SAMPLES_A" \
  --input2 $REF

echo "[B] MedVAE"
fidelity --gpu 0 --fid --samples-find-deep \
  --input1 "$SAMPLES_B" \
  --input2 $REF

echo "[C] MedVAE + REPA"
fidelity --gpu 0 --fid --samples-find-deep \
  --input1 "$SAMPLES_C" \
  --input2 $REF

echo ""
echo "=== Clean-FID & KID ==="

python3 - <<EOF
import torch
from cleanfid import fid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if fid.test_stats_exists("chexpert_train", mode="clean"):
    print("Stats already exist, skipping.")
else:
    fid.make_custom_stats(name="chexpert_train", fdir="$REF", mode="clean", batch_size=64, device=device)

samples = {
    "SD VAE":        "$SAMPLES_A",
    "MedVAE":        "$SAMPLES_B",
    "MedVAE + REPA": "$SAMPLES_C",
}

print(f"\n{'Model':<30} {'Clean-FID':>10} {'KID':>12}")
print("-" * 54)
for name, gen_dir in samples.items():
    score = fid.compute_fid(fdir1=gen_dir, dataset_name="chexpert_train", dataset_res="na", dataset_split="custom", mode="clean", device=device)
    kid   = fid.compute_kid(fdir1=gen_dir, dataset_name="chexpert_train", dataset_res="na", dataset_split="custom", mode="clean", device=device)
    print(f"{name:<30} {score:>10.4f} {kid:>12.6f}")
EOF
