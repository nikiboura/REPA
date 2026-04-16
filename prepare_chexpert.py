"""
prepare_chexpert.py
-------------------
Reads CheXpert images, resizes to 256×256, VAE-encodes, and saves in
the layout expected by Cifar10Dataset:

    <out_dir>/images/<split>/<idx>.png        – resized RGB images
    <out_dir>/vae-sd/<split>/<idx>.npy        – VAE moments (8, h, w) float32
    <out_dir>/vae-sd/<split>/dataset.json     – {"labels": [["<idx>.npy", cls], ...]}

Binary label: 0 = normal ("No Finding" == 1.0), 1 = abnormal
Only frontal views are kept.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


@torch.no_grad()
def encode_batch(vae, imgs, device):
    imgs = imgs.to(device)
    posterior = vae.encode(imgs).latent_dist
    moments = torch.cat([posterior.mean, posterior.std], dim=1)
    return moments.cpu().numpy().astype(np.float32)


def process_split(csv_path, chexpert_root, out_dir, split, resolution,
                  vae, device, batch_size, max_samples=None):
    images_dir = os.path.join(out_dir, 'images', split)
    features_dir = os.path.join(out_dir, 'vae-sd', split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # keep frontal views only
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

    has_no_finding = 'No Finding' in df.columns

    # collect valid (path, label) pairs
    entries = []
    limit = max_samples if max_samples is not None else len(df)
    for i in range(min(limit, len(df))):
        row = df.iloc[i]
        # strip leading folder component from path (e.g. "CheXpert-v1.0-small/")
        img_rel = row['Path'].replace('\\', '/')
        parts = img_rel.split('/', 1)
        img_rel = parts[1] if len(parts) > 1 else parts[0]
        img_path = os.path.join(chexpert_root, img_rel)
        if not os.path.isfile(img_path):
            continue
        nf = row['No Finding'] if has_no_finding else float('nan')
        label = 0 if nf == 1.0 else 1
        entries.append((img_path, label))

    resize_norm = transforms.Compose([
        transforms.Resize((resolution, resolution),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    resize_pil = transforms.Resize(
        (resolution, resolution),
        interpolation=transforms.InterpolationMode.BICUBIC
    )

    labels_meta = []
    for start in tqdm(range(0, len(entries), batch_size), desc=split):
        batch = entries[start:start + batch_size]
        imgs_pil, imgs_tensor, valid_labels = [], [], []
        for img_path, label in batch:
            try:
                img = Image.open(img_path).convert('RGB')
                imgs_pil.append(img)
                imgs_tensor.append(resize_norm(img))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='./data/chexpert_256')
    parser.add_argument('--chexpert-root', type=str, required=True,
                        help='Path to CheXpert-v1.0-small/ (contains train.csv, valid.csv, train/, valid/)')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--vae-model', type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per split (None = all)')
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

    process_split(
        os.path.join(args.chexpert_root, 'train.csv'),
        args.chexpert_root, args.out_dir, 'train',
        args.resolution, vae, device, args.batch_size, args.max_samples
    )
    process_split(
        os.path.join(args.chexpert_root, 'valid.csv'),
        args.chexpert_root, args.out_dir, 'val',
        args.resolution, vae, device, args.batch_size, max_samples=None
    )


if __name__ == '__main__':
    main()
