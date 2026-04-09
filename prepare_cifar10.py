"""
prepare_cifar10.py
------------------
Download CIFAR-10, resize, encode through the
Stable Diffusion VAE and save everything in the
folder layout expected by Cifar10Dataset:

    <out_dir>/images/<split>/<idx>.png          – resized RGB images
    <out_dir>/vae-sd/<split>/<idx>.npy          – VAE moments  (8, h, w) float32
    <out_dir>/vae-sd/<split>/dataset.json       – {"labels": [["<idx>.npy", cls], ...]}
"""

import argparse
import json
import os

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm


@torch.no_grad()
def encode_batch(vae, imgs, device):
    """imgs: float32 tensor (B, 3, H, W) in [-1, 1]"""
    imgs = imgs.to(device)
    posterior = vae.encode(imgs).latent_dist
    # return concatenated mean & std (moments) — shape (B, 8, H/8, W/8)
    moments = torch.cat([posterior.mean, posterior.std], dim=1)
    return moments.cpu().numpy().astype(np.float32)


def process_split(dataset, out_dir, split, resolution, vae, device, batch_size):
    images_dir = os.path.join(out_dir, 'images', split)
    features_dir = os.path.join(out_dir, 'vae-sd', split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    resize = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),              # [0, 1]
        transforms.Normalize([0.5]*3, [0.5]*3),  # [-1, 1]  for VAE
    ])
    resize_pil = transforms.Resize(
        (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC
    )

    labels_meta = []
    n = len(dataset)

    for start in tqdm(range(0, n, batch_size), desc=f'{split}'):
        end = min(start + batch_size, n)
        imgs_tensor = []
        for i in range(start, end):
            img_pil, _ = dataset[i]
            imgs_tensor.append(resize(img_pil))
        imgs_tensor = torch.stack(imgs_tensor)

        moments = encode_batch(vae, imgs_tensor, device)  # (B, 8, h, w)

        for j, i in enumerate(range(start, end)):
            img_pil, label = dataset[i]
            img_pil = resize_pil(img_pil)
            fname = f'{i}.png'
            img_pil.save(os.path.join(images_dir, fname))

            npy_fname = f'{i}.npy'
            np.save(os.path.join(features_dir, npy_fname), moments[j])
            labels_meta.append([npy_fname, int(label)])

    with open(os.path.join(features_dir, 'dataset.json'), 'w') as f:
        json.dump({'labels': labels_meta}, f)

    print(f'[{split}] saved {n} images and VAE latents to {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='./data/cifar10_256')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--vae-model', type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument('--cifar-root', type=str, default='./data/cifar10_raw')
    args = parser.parse_args()

    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {device}')

    print('Loading VAE...')
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    vae.eval()

    for split, train_flag in [('train', True), ('val', False)]:
        dataset = CIFAR10(root=args.cifar_root, train=train_flag, download=True, transform=None)
        process_split(dataset, args.out_dir, split, args.resolution, vae, device, args.batch_size)


if __name__ == '__main__':
    main()
