"""
compute_metrics.py
-------------------
Evaluation metrics.

Usage:
    python compute_metrics.py \
        --data-dir /content/data/chexpert_256_sdvae \
        --generated-dir /content/generated_pe \
        --split train
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import wandb
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset import LatentDataset

from cleanfid import fid
from diffusers.models import AutoencoderKL
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


def build_real_class_dir(data_dir, split, label, tmp_root):
    """Symlink real images of a given class into a fresh temp dir for clean-fid."""
    dataset = LatentDataset(data_dir, split=split)
    out_dir = os.path.join(tmp_root, f'real_class_{label}')
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for fname, lbl in dataset.entries:
        if lbl != label:
            continue
        base = os.path.splitext(fname)[0]
        src = os.path.join(dataset.images_dir, base + '.png')
        dst = os.path.join(out_dir, base + '.png')
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
        n += 1
    return out_dir, n


def build_generated_pe_dir(generated_dir, tmp_root):
    """Symlink only the *_pe.png files (skip *_healthy.png) into a fresh temp dir."""
    out_dir = os.path.join(tmp_root, 'generated_pe')
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for fname in os.listdir(generated_dir):
        if fname.endswith('_pe.png'):
            src = os.path.join(generated_dir, fname)
            dst = os.path.join(out_dir, fname)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
            n += 1
    return out_dir, n


class LatentClassifier(nn.Module):
    """Small CNN over VAE latent means -- Healthy (0) vs PE (1)."""

    def __init__(self, in_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc(h).squeeze(-1)


def compute_auc(scores, labels):
    """Rank-based AUC (Mann-Whitney U statistic) -- no sklearn dependency."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def latent_means(dataset, indices):
    xs, ys = [], []
    for i in indices:
        _, moments, label = dataset[i]
        mean, _ = torch.chunk(moments, 2, dim=0)
        xs.append(mean)
        ys.append(label.item())
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)


def compute_pair_similarity(generated_dir, resolution, device):
    """SSIM/PSNR/LPIPS between each {stem}_healthy.png and {stem}_pe.png pair."""
    stems = sorted(
        fname[:-len('_healthy.png')] for fname in os.listdir(generated_dir)
        if fname.endswith('_healthy.png')
        and os.path.isfile(os.path.join(generated_dir, fname[:-len('_healthy.png')] + '_pe.png'))
    )

    lpips_fn = lpips.LPIPS(net='alex').to(device).eval() if HAS_LPIPS else None
    if lpips_fn is None:
        print('  [warning] lpips not installed (`pip install lpips`) -- skipping LPIPS.')

    ssim_scores, psnr_scores, lpips_scores = [], [], []
    for stem in stems:
        healthy = np.array(
            Image.open(os.path.join(generated_dir, f'{stem}_healthy.png')).convert('RGB').resize((resolution, resolution))
        )
        pe = np.array(
            Image.open(os.path.join(generated_dir, f'{stem}_pe.png')).convert('RGB').resize((resolution, resolution))
        )
        ssim_scores.append(ssim_metric(healthy, pe, channel_axis=-1, data_range=255))
        psnr_scores.append(psnr_metric(healthy, pe, data_range=255))
        if lpips_fn is not None:
            h_t = torch.from_numpy(healthy).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)
            p_t = torch.from_numpy(pe).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)
            with torch.no_grad():
                lpips_scores.append(lpips_fn(h_t, p_t).item())

    return {
        'n_pairs': len(stems),
        'ssim': float(np.mean(ssim_scores)) if ssim_scores else float('nan'),
        'psnr': float(np.mean(psnr_scores)) if psnr_scores else float('nan'),
        'lpips': float(np.mean(lpips_scores)) if lpips_scores else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--generated-dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report-to', type=str, default='none', choices=['none', 'wandb'])
    parser.add_argument('--wandb-project', type=str, default='REPA')
    parser.add_argument('--wandb-name', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    if args.report_to == 'wandb':
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    tmp_root = tempfile.mkdtemp(prefix='cf_metrics_')
    try:
        # ---- FID / KID: generated PE vs real PE ----
        real_pe_dir, n_real_pe = build_real_class_dir(args.data_dir, args.split, 1, tmp_root)
        gen_pe_dir, n_gen_pe = build_generated_pe_dir(args.generated_dir, tmp_root)
        print(f'Real PE images: {n_real_pe}, Generated PE images: {n_gen_pe}')
        assert n_real_pe > 0, f'No PE images found in {args.data_dir} split={args.split}'
        assert n_gen_pe > 0, f'No *_pe.png files found in {args.generated_dir}'

        print('\nComputing FID...')
        fid_score = fid.compute_fid(fdir1=gen_pe_dir, fdir2=real_pe_dir, mode='clean', device=device)
        print('Computing KID...')
        kid_score = fid.compute_kid(fdir1=gen_pe_dir, fdir2=real_pe_dir, mode='clean', device=device)

        # ---- AUC: train a Healthy-vs-PE classifier on real latents ----
        print('\nTraining pathology classifier on real latents...')
        dataset = LatentDataset(args.data_dir, split=args.split)
        idx = np.arange(len(dataset))
        rng = np.random.RandomState(args.seed)
        rng.shuffle(idx)
        n_val = int(len(dataset) * args.val_frac)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train, y_train = latent_means(dataset, train_idx)
        X_val, y_val = latent_means(dataset, val_idx)

        clf = LatentClassifier(in_channels=X_train.shape[1]).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()

        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)

        for epoch in range(args.epochs):
            clf.train()
            perm = torch.randperm(len(X_train))
            total_loss = 0.
            for start in range(0, len(X_train), args.batch_size):
                b = perm[start:start + args.batch_size]
                opt.zero_grad()
                loss = loss_fn(clf(X_train[b]), y_train[b])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(b)
            print(f'  epoch {epoch + 1}/{args.epochs}  loss={total_loss / len(X_train):.4f}')

        clf.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(clf(X_val)).cpu().numpy()
        real_auc = compute_auc(val_scores, y_val.cpu().numpy())

        # ---- Score generated PE images with the same classifier ----
        print('\nEncoding generated PE images through the VAE for classifier scoring...')
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(device).eval()
        to_tensor = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        gen_files = sorted(f for f in os.listdir(gen_pe_dir) if f.endswith('.png'))
        gen_means = []
        with torch.no_grad():
            for start in range(0, len(gen_files), args.batch_size):
                batch_files = gen_files[start:start + args.batch_size]
                imgs = torch.stack([
                    to_tensor(Image.open(os.path.join(gen_pe_dir, f)).convert('RGB'))
                    for f in batch_files
                ]).to(device)
                gen_means.append(vae.encode(imgs).latent_dist.mean.cpu())
        gen_means = torch.cat(gen_means).to(device)

        # Real Healthy held-out latents = the "known non-disease" comparison set
        real_healthy_val = X_val[y_val == 0]
        eval_X = torch.cat([real_healthy_val, gen_means])
        eval_y = np.concatenate([np.zeros(len(real_healthy_val)), np.ones(len(gen_means))])
        with torch.no_grad():
            eval_scores = torch.sigmoid(clf(eval_X)).cpu().numpy()
        generated_auc = compute_auc(eval_scores, eval_y)

        # ---- Pair similarity: Healthy source vs its generated PE counterfactual ----
        print('\nComputing Healthy vs generated-PE pair similarity...')
        pair_metrics = compute_pair_similarity(args.generated_dir, args.resolution, device)

        # ---- Report ----
        print()
        print('FID:', fid_score)
        print('KID:', kid_score)
        print('AUC:', generated_auc)
        print('SSIM (healthy vs counterfactual):', pair_metrics['ssim'])
        print('PSNR (healthy vs counterfactual):', pair_metrics['psnr'])
        print('LPIPS (healthy vs counterfactual):', pair_metrics['lpips'])

        if args.report_to == 'wandb':
            wandb.log({
                'fid': fid_score,
                'kid': kid_score,
                'auc': generated_auc,
                'ssim': pair_metrics['ssim'],
                'psnr': pair_metrics['psnr'],
                'lpips': pair_metrics['lpips'],
            })

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        if args.report_to == 'wandb':
            wandb.finish()


if __name__ == '__main__':
    main()
