<h1 align="center"> Medical Image Counterfactual Generation via
Vision Foundation-Guided Diffusion Models
</h1>


<b>Summary</b>: Repo is forked from https://github.com/sihyun-yu/REPA where Representation Alignment method is introduced to align noisy input states in diffusion models with representations from pretrained visual encoders. Using this method we want to extend the model to editing cababilities for counterfactual generation. 

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
A Kaggle API token is required to download the CheXpert dataset and MedDINOv3 weights during preprocessing and training. Create from https://www.kaggle.com/settings/api and note the path.

### 3. Download pre-trained weights from SiT/XL/2 model
The following checkpoint is used during training to initialize weights from SiT/XL/2 model using this implementation https://github.com/OscarXZQ/weight-selection.

You can use the following link to download it: https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0

The link was retrieved from utils.py

### 2. Dataset

#### Dataset download & preprocessing
  To preprocess the dataset (resize to 256×256 and encode with VAE), set your
  `KAGGLE_JSON` path in `prepare_data.sh` and run:

  ```bash
  # 80k samples
  bash prepare_data.sh
  
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

You can generate images (and the .npz file can be used for [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations) suite) through the following script:

```bash
torchrun --nnodes=1 --nproc_per_node=8 generate.py \
  --model SiT-XL/2 \
  --num-fid-samples 50000 \
  --ckpt YOUR_CHECKPOINT_PATH \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.8 \
  --guidance-high=0.7
```

We also provide the SiT-XL/2 checkpoint (trained for 4M iterations) used in the final evaluation. It will be automatically downloaded if you do not specify `--ckpt`.

### Note

It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. Additionally, we'll make an effort to carry out sanity-check experiments in the near future.

## Acknowledgement

This code is mainly built upon [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT), [edm2](https://github.com/NVlabs/edm2), and [RCG](https://github.com/LTH14/rcg) repositories.\
We also appreciate [Kyungmin Lee](https://kyungmnlee.github.io/) for providing the initial version of the implementation.

## BibTeX

```bibtex
@inproceedings{yu2025repa,
  title={Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think},
  author={Sihyun Yu and Sangkyung Kwak and Huiwon Jang and Jongheon Jeong and Jonathan Huang and Jinwoo Shin and Saining Xie},
  year={2025},
  booktitle={International Conference on Learning Representations},
}
```
