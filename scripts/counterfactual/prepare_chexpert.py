"""
prepare_chexpert.py
-------------------
Reads Chexpert images, resizes to 256×256, VAE-encodes, and saves in
the layout expected.

Output:
    <out_dir>/images/<split>/<idx>.png        – resized RGB images
    <out_dir>/vae-sd/<split>/<idx>.npy        – VAE moments (8, h, w) float32
    <out_dir>/vae-sd/<split>/dataset.json     – {"labels": [["<idx>.npy", cls],...]}

Usage:
python /content/REPA/scripts/counterfactual/prepare_chexpert.py \
      --chexpert-root /content/chexpert \
      --out-dir /content/data/chexpert_256_sdvae \
      --resolution 256 \
      --pathologies "Pleural Effusion" \
      --vae-type sd \
      --test-size 500


Labels:
    0           = healthy  (No Finding == 1.0, regardless of any other column)
    1 .. N      = each pathology passed via --pathologies, in order

A pathology row only keeps a label when it is positive and every other
disease column is negative (Support Devices is ignored, it isn't a
disease). Classes are downsampled to equal size.
"""

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

DISEASE_COLS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
]


def load_vae(vae_type, device):
    if vae_type == 'medvae':
        from medvae import MVAE
        return MVAE(model_name='medvae_8_4_2d', modality='xray').model.to(device).eval()
    return AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(device).eval()


@torch.no_grad()
def encode_batch(vae, imgs, device):
    imgs = imgs.to(device)
    result = vae.encode(imgs)
    posterior = result.latent_dist if hasattr(result, 'latent_dist') else result
    moments = torch.cat([posterior.mean, posterior.std], dim=1)
    return moments.cpu().numpy().astype(np.float32)


def build_labels(df, pathologies):
    """
      0    = healthy, No Finding == 1.0, unconditionally, no other column checked.
      i    = pathologies[i-1] is positive and every other disease column negative.
      NaN  = neither condition met row is dropped
    """
    disease_cols = [c for c in DISEASE_COLS if c in df.columns]
    label = pd.Series(np.nan, index=df.index)

    if 'No Finding' in df.columns:
        label[df['No Finding'] == 1.0] = 0

    for i, path in enumerate(pathologies, start=1):
        if path not in df.columns:
            continue
        others = [c for c in disease_cols if c != path]
        pure = (df[path] == 1.0) & (df[others] != 1.0).all(axis=1)
        label[label.isna() & pure] = i

    return label


def split_test_rows(df, pathologies, n_test=500, seed=42):
    """
    Build a test split: for each label (Healthy and every pathology),
    randomly sample exactly n_test rows (or every available row, if fewer
    than n_test exist for that label). Everything not sampled stays in train.

    Both returned dataframes are already labeled (via build_labels) and
    filtered (NaN/co-morbid/uncertain rows dropped).
    """
    labeled = df.assign(label=build_labels(df, pathologies)).dropna(subset=['label'])

    test_parts = []
    for label in range(len(pathologies) + 1):  # 0 = Healthy, 1..N = pathologies
        pool = labeled[labeled['label'] == label]
        test_parts.append(pool.sample(n=min(n_test, len(pool)), random_state=seed))

    test_df = pd.concat(test_parts)
    train_df = labeled.drop(index=test_df.index)
    return train_df, test_df


def balance_classes(df, seed=42):
    """Downsample every class to the size of the rarest one, then shuffle using frac=1."""
    min_count = df['label'].value_counts().min()
    balanced = df.groupby('label', group_keys=False).sample(n=min_count, random_state=seed)
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def _load_resized(path, resize):
    try:
        return resize(Image.open(path).convert('RGB'))
    except Exception:
        return None


def _print_class_counts(df, label_names):
    counts = df['label'].value_counts().reindex(range(len(label_names)), fill_value=0)
    for idx, name in enumerate(label_names):
        print(f'  {idx} ({name}): {counts[idx]}')
    print(f'  Total: {counts.sum()}')


def process_split(df, chexpert_root, out_dir, split, resolution,
                  vae, device, batch_size, pathologies, max_samples=None):
    images_dir = os.path.join(out_dir, 'images', split)
    features_dir = os.path.join(out_dir, 'vae-sd', split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    df = df.copy()
    if max_samples is not None:
        df = df.head(max_samples)

    df['label'] = df['label'].astype(int)

    # resolve image path (strip leading "CheXpert-v1.0-small/" folder component)
    # and drop rows whose file isn't actually on disk
    rel_path = df['Path'].str.replace('\\', '/', regex=False).str.split('/', n=1).str[-1]
    df['image_path'] = rel_path.apply(lambda p: os.path.join(chexpert_root, p))
    df = df[df['image_path'].apply(os.path.isfile)]

    label_names = ['Healthy'] + list(pathologies)
    print(f'\n[{split}] class distribution before balancing:')
    _print_class_counts(df, label_names)

    df = balance_classes(df[['image_path', 'label']])

    print(f'\n[{split}] class distribution after balancing:')
    _print_class_counts(df, label_names)

    resize = transforms.Resize((resolution, resolution),
                               interpolation=transforms.InterpolationMode.BICUBIC)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    labels_meta = []
    rows = df.to_dict('records')
    for start in tqdm(range(0, len(rows), batch_size), desc=split):
        batch = rows[start:start + batch_size]
        imgs, labels = [], []
        for row in batch:
            img = _load_resized(row['image_path'], resize)
            if img is None:
                continue
            imgs.append(img)
            labels.append(row['label'])

        if not imgs:
            continue

        tensors = torch.stack([to_tensor(img) for img in imgs])
        moments = encode_batch(vae, tensors, device)

        for img, label, moment in zip(imgs, labels, moments):
            idx = len(labels_meta)
            img.save(os.path.join(images_dir, f'{idx}.png'))
            np.save(os.path.join(features_dir, f'{idx}.npy'), moment)
            labels_meta.append([f'{idx}.npy', label])

    with open(os.path.join(features_dir, 'dataset.json'), 'w') as f:
        json.dump({'labels': labels_meta}, f)

    print(f'[{split}] saved {len(labels_meta)} images and VAE latents to {out_dir}')
    return len(labels_meta)


def download_chexpert(chexpert_root, kaggle_json=None):
    if os.path.isfile(os.path.join(chexpert_root, 'train.csv')):
        print('CheXpert already downloaded, skipping.')
        return
    if kaggle_json and os.path.isfile(kaggle_json):
        kaggle_dir = os.path.expanduser('~/.kaggle')
        os.makedirs(kaggle_dir, exist_ok=True)
        dest = os.path.join(kaggle_dir, 'kaggle.json')
        shutil.copy(kaggle_json, dest)
        os.chmod(dest, 0o600)
    os.makedirs(chexpert_root, exist_ok=True)
    print('Downloading CheXpert from Kaggle')
    subprocess.run(
        ['kaggle', 'datasets', 'download', 'ashery/chexpert',
         '-p', chexpert_root, '--unzip'],
        check=True,
    )
    print('Download complete.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='./data/chexpert_256')
    parser.add_argument('--chexpert-root', type=str, required=True,
                        help='Path to CheXpert-v1.0-small/ (contains train.csv)')
    parser.add_argument('--kaggle-json', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--vae-type', type=str, default='sd', choices=['sd', 'medvae'])
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max rows to scan per split (None = all)')
    parser.add_argument('--pathologies', nargs='+',
                        default=['Pleural Effusion'],
                        help='Target pathology names (single-label rows only). '
                             'E.g. --pathologies "Pleural Effusion" "Cardiomegaly"')
    parser.add_argument('--test-size', type=int, default=500,
                        help='Number of rows to sample per label (Healthy and each pathology) '
                             'for the test split.')
    args = parser.parse_args()

    download_chexpert(args.chexpert_root, args.kaggle_json)

    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {device}')
    print(f'Target pathologies: {args.pathologies}')

    print('Loading VAE...')
    vae = load_vae(args.vae_type, device)

    df = pd.read_csv(os.path.join(args.chexpert_root, 'train.csv'))
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']

    train_df, test_df = split_test_rows(df, args.pathologies, n_test=args.test_size)
    print(f'\nSampled up to {args.test_size} rows per label for testing -- removed from training.')

    n_train = process_split(
        train_df, args.chexpert_root, args.out_dir, 'train',
        args.resolution, vae, device, args.batch_size,
        args.pathologies, args.max_samples,
    )
    n_test = process_split(
        test_df, args.chexpert_root, args.out_dir, 'test',
        args.resolution, vae, device, args.batch_size,
        args.pathologies, max_samples=None,
    )

    print(f'\n=== Dataset summary ===')
    print(f'  Training samples: {n_train}')
    print(f'  Testing samples:  {n_test}')
    print(f'  Total:            {n_train + n_test}')


if __name__ == '__main__':
    main()
