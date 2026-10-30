<h1 align="center"> Medical Image Counterfactual Generation via
Vision Foundation-Guided Diffusion Models
</h1>

<b>Summary</b>: Repo is forked from https://github.com/sihyun-yu/REPA where Representation Alignment (REPA) is introduced to align noisy diffusion states with representations from pretrained visual encoders. We extend this with editing capabilities for counterfactual generation (Healthy → Pleural Effusion) on CheXpert.

There are two separate experiment pipelines in this repo:
- **`run/model_ablations/`** — VAE choice × REPA ablation (does REPA help, does VAE choice matter).
- **`run/produce_counterfactuals/`** — REPA × cond-x1 counterfactual generation (Healthy → Pleural Effusion).

## Setup

### 1. Clone repo
```bash
git clone https://github.com/nikiboura/REPA.git
cd REPA
```

### 2. Environment
```bash
bash run/model_ablations/install.sh
```
(Same install script works for both pipelines.)

### 3. Kaggle token
Needed to download CheXpert and MedDINOv3 weights. Create one at https://www.kaggle.com/settings/api and note the path — you'll set it as `KAGGLE_JSON` in each `prepare_data.sh`.

### 4. Teacher checkpoint
Used for weight-selection initialization from SiT-XL/2 (https://github.com/OscarXZQ/weight-selection). Download: https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0

Set this as `TEACHER_CKPT` in each `train.sh`.

---

## Ablations (`run/model_ablations/`)

Four experiments isolating VAE choice and REPA, independently:

| | VAE | REPA |
|---|---|---|
| A | SD | off |
| B | MedVAE | off |
| C | MedVAE | on |
| D | SD | on |

```bash
# 1. Prepare data (set KAGGLE_JSON first)
bash run/model_ablations/prepare_data.sh        # 80k samples
bash run/model_ablations/prepare_data.sh -1     # full dataset
bash run/model_ablations/prepare_data.sh -s     # sanity check

# 2. Train all four experiments (set TEACHER_CKPT first)
bash run/model_ablations/train.sh               # SiT-S/4, 400k steps
bash run/model_ablations/train.sh -1             # SiT-B/2, 4M steps
bash run/model_ablations/train.sh -s             # sanity check

# 3. Generate samples
bash run/model_ablations/generate.sh
bash run/model_ablations/generate.sh -1
bash run/model_ablations/generate.sh -s

# 4. Compute metrics (FID, Clean-FID, KID)
bash run/model_ablations/compute_metrics.sh
```

---

## Counterfactuals (`run/produce_counterfactuals/`)

Four experiments isolating REPA and `--cond-x1` (source-latent conditioning), independently, for the Healthy → Pleural Effusion bridge:

| | REPA | cond-x1 |
|---|---|---|
| A | on | off |
| B | on | on |
| C | off | off |
| D | off | on |

```bash
# 1. Prepare data (set KAGGLE_JSON first)
bash run/produce_counterfactuals/prepare_data.sh

# 2. Train all four experiments (set TEACHER_CKPT first)
bash run/produce_counterfactuals/train.sh        # SiT-S/4, 400k steps
bash run/produce_counterfactuals/train.sh -1     # SiT-B/2
bash run/produce_counterfactuals/train.sh -s     # sanity check

# 3. Generate counterfactuals (Healthy → Pleural Effusion)
bash run/produce_counterfactuals/generate.sh
bash run/produce_counterfactuals/generate.sh -s

# 4. Compute metrics (FID, KID, AUC, SSIM, PSNR, LPIPS)
bash run/produce_counterfactuals/compute_metrics.sh
```

---

## UNSB baseline (`run/unsb_baseline/`)

[UNSB](https://github.com/cyclomon/UNSB) (Unpaired Neural Schrödinger Bridge, ICLR 2024) solves the
same unpaired Healthy → Pleural Effusion translation problem as `produce_counterfactuals/`, but via
an adversarially-trained Schrödinger Bridge instead of our REPA/I2SB diffusion pipeline. Used here as
a baseline for comparison. The official repo is vendored as a submodule at `third_party/UNSB`.

```bash
# 0. Fetch the UNSB submodule + its extra deps (dominate, visdom, gputil, packaging)
bash run/unsb_baseline/install.sh

# 1. Re-sort the existing chexpert_256_sdvae images into UNSB's trainA/trainB/testA/testB layout
#    (requires run/produce_counterfactuals/prepare_data.sh to have been run first)
bash run/unsb_baseline/prepare_data.sh

# 2. Train UNSB (Healthy -> Pleural Effusion, mode=sb, N=5 refinement steps)
bash run/unsb_baseline/train.sh
bash run/unsb_baseline/train.sh -s     # sanity check

# 3. Generate counterfactuals at NFE=5 and reshape them to match compute_metrics.py's
#    <idx>_healthy.png / <idx>_pe.png convention
bash run/unsb_baseline/generate.sh
bash run/unsb_baseline/generate.sh -s

# 4. Compute the same metrics (FID, KID, AUC, SSIM, PSNR, LPIPS) as produce_counterfactuals/,
#    directly comparable across pipelines
bash run/unsb_baseline/compute_metrics.sh
```
