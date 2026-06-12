<h1 align="center"> Medical Image Counterfactual Generation via
Vision Foundation-Guided Diffusion Models
</h1>


<b>Summary</b>: Repo is forked from https://github.com/sihyun-yu/REPA where Representation Alignment method is introduced to align noisy input states in diffusion models with representations from pretrained visual encoders. Using this method we want to extend the model with editing cababilities for counterfactual generation. The following instructions cover the preliminary experiments.

### 1. Clone repo
```bash
git clone https://github.com/nikiboura/REPA.git
cd REPA
```
### 1. Environment setup

```bash
conda create -n repa python=3.9 -y
conda activate repa
bash install.sh
```

### 3. Kaggle Token
A Kaggle API token is required to download the CheXpert dataset and MedDINOv3 weights during preprocessing and training. 

- Create from https://www.kaggle.com/settings/api and note the path.

### 3. Download pre-trained weights from SiT/XL/2 model
The following checkpoint is used during training to initialize weights from SiT/XL/2 model from this implementation https://github.com/OscarXZQ/weight-selection.

- Use the following link to download it: https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0

The link was retrieved from utils.py

### 2. Dataset

#### Dataset download & preprocessing
  To preprocess the dataset (resize to 256×256 and encode with VAE), set your
  `KAGGLE_JSON` path in `prepare_data.sh` and run:

  ```bash
  # 80k samples
  bash prepare_data.sh
  ```

  ```bash
  # full dataset
  bash prepare_data.sh -1
  ```

  This will create two preprocessed directories:
  - ./data/chexpert_256_sdvae — encoded with the standard Stable Diffusion VAE
  (Experiment A)

  - ./data/chexpert_256_medvae — encoded with MedVAE (Experiments B & C)

### 3. Training
Set your `TEACHER_CKPT` path in `train.sh`, then run:
  
```bash
# Ablation: SiT-S/4, 400k steps
bash train.sh
```

```bash
# Full training: SiT-B/2, 4M steps
bash train.sh -1
 ```
This runs all three experiments sequentially:
  - A — SiT + SD VAE (no REPA)
  - B — SiT + MedVAE (no REPA)
  - C — SiT + MedVAE + REPA alignment (MedDINOv3 downloads automatically)
  
The teacher checkpoint (last.pt) is the pretrained SiT-XL/2 model (4M iterations)
from the original REPA repo, used to initialize student weights for faster
convergence. 
 
### 4. Evaluation
Run generation for all three experiments:
  
```bash
# Ablation: SiT-S/4, 400k checkpoint, 50k samples
bash generate.sh
```

```bash
# Full: SiT-B/2, 4M checkpoint, 50k samples
bash generate.sh -1
```

Generated samples are saved as .png files and a .npz file under ./results/<exp-name>/samples/.
