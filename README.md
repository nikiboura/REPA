<h1 align="center"> Medical Image Counterfactual Generation via
Vision Foundation-Guided Diffusion Models
</h1>

<b>Summary</b>: Repo is forked from https://github.com/sihyun-yu/REPA where Representation Alignment (REPA) is introduced to align noisy diffusion states with representations from pretrained visual encoders. We extend this with editing capabilities for counterfactual generation (Healthy → Pleural Effusion) on CheXpert.

There are three separate experiment pipelines in this repo:
- **`run/model_ablations/`** — VAE choice × REPA ablation (does REPA help, does VAE choice matter).
- **`run/produce_counterfactuals/`** — REPA × cond-x1 counterfactual generation (Healthy → Pleural Effusion).
- **`run/unsb_baseline/`** — UNSB, an external unpaired-translation baseline for comparison.

## Setup

### 1. Clone repo
```bash
git clone https://github.com/nikiboura/REPA.git
cd REPA
```

### 2. Environment
```bash
bash install.sh
conda activate repa
```
One script for all three pipelines — conda env, Python deps, and the UNSB submodule + its extra deps.
`conda activate repa` is needed again in every new terminal session (the env itself persists, the activation doesn't).

On a shared machine with a small home-directory quota, redirect the conda env and all package/model
caches to a bigger disk instead (and clone/run this repo itself from inside that same folder too):
```bash
export SCRATCH_DIR=/path/to/scratch
cd $SCRATCH_DIR
git clone https://github.com/nikiboura/REPA.git
cd REPA
bash install.sh
```
`install.sh` prints the exact `conda activate ...` command to use afterward — it won't be `conda activate repa` in this case.

### 3. Kaggle token
Needed to download CheXpert and MedDINOv3 weights. Create one at https://www.kaggle.com/settings/api, then:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
(if your browser saved it somewhere other than `~/Downloads`, adjust the source path above)

Every `prepare_data.sh` in this repo defaults to `~/.kaggle/kaggle.json` — nothing else to set, unless your token lives somewhere else, in which case: `export KAGGLE_JSON=/path/to/kaggle.json` before running `prepare_data.sh`.

### 4. Teacher checkpoint
Used for weight-selection initialization from SiT-XL/2 (https://github.com/OscarXZQ/weight-selection). Download: https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0

Only needed for `model_ablations/` and `produce_counterfactuals/` (not `unsb_baseline/`). Before running either's `train.sh`:
```bash
export TEACHER_CKPT=/path/to/last.pt
```
(`train.sh` errors out immediately with a clear message if this isn't set — no silent failures.)

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
# 1. Prepare data (uses ~/.kaggle/kaggle.json by default, see Setup above)
# no flag: 80k samples, -1: full dataset, -s: sanity check
bash run/model_ablations/prepare_data.sh
bash run/model_ablations/prepare_data.sh -1
bash run/model_ablations/prepare_data.sh -s

# 2. Train all four experiments (set TEACHER_CKPT first)
# no flag: SiT-S/4, 400k steps, -1: SiT-L/2, 4M steps, -s: sanity check
bash run/model_ablations/train.sh
bash run/model_ablations/train.sh -1
bash run/model_ablations/train.sh -s

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
# 1. Prepare data (uses ~/.kaggle/kaggle.json by default, see Setup above)
bash run/produce_counterfactuals/prepare_data.sh

# 2. Train all four experiments (set TEACHER_CKPT first)
# no flag: SiT-S/4, 400k steps, -1: SiT-L/2, -s: sanity check
bash run/produce_counterfactuals/train.sh
bash run/produce_counterfactuals/train.sh -1
bash run/produce_counterfactuals/train.sh -s

# 3. Generate counterfactuals (Healthy → Pleural Effusion)
bash run/produce_counterfactuals/generate.sh
bash run/produce_counterfactuals/generate.sh -1
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

UNSB trains in pixel space and never reads VAE latents, so this pipeline builds its own
`chexpert_256_unsb` split (`--skip-vae`: resized PNGs + labels, no SD-VAE download/encoding) for
training.
`compute_metrics.sh` is the one that still needs `chexpert_256_sdvae` (built separately by
`run/produce_counterfactuals/prepare_data.sh`) as its VAE latents are what the AUC metric needs.

```bash
# (the UNSB submodule + its extra deps are already pulled by the root install.sh)

# 1. Build a lean Healthy/Pleural-Effusion split (no VAE) + UNSB's trainA/trainB/testA/testB layout
bash run/unsb_baseline/prepare_data.sh

# 2. Train UNSB (Healthy -> Pleural Effusion, mode=sb, N=5 refinement steps, 400 epochs)
# -s: sanity check (1 epoch, skips resume logic)
bash run/unsb_baseline/train.sh
bash run/unsb_baseline/train.sh -s

# 3. Generate counterfactuals at NFE=5 and reshape them to match compute_metrics.py's
# <idx>_healthy.png / <idx>_pe.png convention (-s: sanity check)
bash run/unsb_baseline/generate.sh
bash run/unsb_baseline/generate.sh -s

# 4. Compute metrics (FID, KID, SSIM, PSNR, LPIPS, AUC)
bash run/unsb_baseline/compute_metrics.sh
```
