"""
prepare_imagenet256.py
----------------------
Reads ImageNet 256×256 images and VAE-encodes them, saving in the layout
expected.

    <out_dir>/images/<split>/<idx>.png        – RGB images
    <out_dir>/vae-sd/<split>/<idx>.npy        – VAE moments (8, h, w) float32
    <out_dir>/vae-sd/<split>/dataset.json     – {"labels": [["<idx>.npy", cls], ...]}

Supports two folder layouts (auto-detected):
  A) <root>/train/<class>/img.jpg  +  <root>/val/<class>/img.jpg
  B) <root>/<class>/img.jpg  (no train/val split — auto split 90/10)
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


@torch.no_grad()
def encode_batch(vae, imgs, device):
    imgs = imgs.to(device)
    posterior = vae.encode(imgs).latent_dist
    moments = torch.cat([posterior.mean, posterior.std], dim=1)
    return moments.cpu().numpy().astype(np.float32)


def process_entries(entries, out_dir, split, resolution, vae, device, batch_size):
    """entries: list of (img_path, class_idx)"""
    images_dir = os.path.join(out_dir, 'images', split)
    features_dir = os.path.join(out_dir, 'vae-sd', split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    to_tensor_norm = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    resize_pil = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC)

    labels_meta = []
    for start in tqdm(range(0, len(entries), batch_size), desc=split):
        batch = entries[start:start + batch_size]
        imgs_pil, imgs_tensor, valid_labels = [], [], []
        for img_path, label in batch:
            try:
                img = Image.open(img_path).convert('RGB')
                imgs_pil.append(img)
                imgs_tensor.append(to_tensor_norm(img))
                valid_labels.append(label)
            except Exception:
                continue

        if not imgs_tensor:
            continue

        moments = encode_batch(vae, torch.stack(imgs_tensor), device)

        for k, (img_pil, label) in enumerate(zip(imgs_pil, valid_labels)):
            idx = len(labels_meta)
            resize_pil(img_pil).save(os.path.join(images_dir, f'{idx}.png'))
            npy_fname = f'{idx}.npy'
            np.save(os.path.join(features_dir, npy_fname), moments[k])
            labels_meta.append([npy_fname, label])

    with open(os.path.join(features_dir, 'dataset.json'), 'w') as f:
        json.dump({'labels': labels_meta}, f)

    print(f'[{split}] saved {len(labels_meta)} images and VAE latents to {out_dir}')


def get_entries_from_imagefolder(folder):
    ds = ImageFolder(folder)
    return [(path, label) for path, label in ds.samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='./data/imagenet256')
    parser.add_argument('--imagenet-root', type=str, required=True,
                        help='Root folder of the ImageNet-256 dataset')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--vae-model', type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument('--max-train-samples', type=int, default=5000)
    parser.add_argument('--max-val-samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {device}')

    # Auto-detect layout
    has_train = os.path.isdir(os.path.join(args.imagenet_root, 'train'))
    has_val   = os.path.isdir(os.path.join(args.imagenet_root, 'val'))

    if has_train and has_val:
        print('Detected layout: train/ + val/ subfolders')
        train_entries = get_entries_from_imagefolder(os.path.join(args.imagenet_root, 'train'))
        val_entries   = get_entries_from_imagefolder(os.path.join(args.imagenet_root, 'val'))
    else:
        print('Detected layout: flat class folders — creating 90/10 split')
        all_entries = get_entries_from_imagefolder(args.imagenet_root)
        random.shuffle(all_entries)
        split_idx = int(len(all_entries) * 0.9)
        train_entries = all_entries[:split_idx]
        val_entries   = all_entries[split_idx:]

    # Subsample
    if args.max_train_samples:
        random.shuffle(train_entries)
        train_entries = train_entries[:args.max_train_samples]
    if args.max_val_samples:
        random.shuffle(val_entries)
        val_entries = val_entries[:args.max_val_samples]

    print(f'Train: {len(train_entries)} | Val: {len(val_entries)}')

    print('Loading VAE...')
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    vae.eval()

    process_entries(train_entries, args.out_dir, 'train', args.resolution, vae, device, args.batch_size)
    process_entries(val_entries,   args.out_dir, 'val',   args.resolution, vae, device, args.batch_size)


if __name__ == '__main__':
    main()
