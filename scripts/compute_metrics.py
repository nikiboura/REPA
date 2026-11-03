"""
compute_metrics.py
-------------------
Evaluation metrics for the model_ablations pipeline: how close generated
CheXpert samples look to real CheXpert training images. No pairing or
classification involved here -- unlike scripts/counterfactual/compute_metrics.py,
which scores the Healthy -> Pleural Effusion counterfactual task specifically,
this is unconditional/class-conditional generation quality only.

Reports two independent FID numbers, because different libraries' image-resizing
pipelines shift the score enough that they aren't directly comparable:
  - torch-fidelity's FID -- a common baseline-reported value.
  - clean-fid's FID -- a standardized reimplementation (CVPR 2022, "On Aliasing,
    Rasterization, and Image Resizing...") used as the modern comparability
    convention in a lot of recent generative-modeling papers.
KID is only computed via clean-fid (torch-fidelity's FID-only call here doesn't
compute it).

Usage:
    python scripts/compute_metrics.py \
        --ref ./data/chexpert_256_sdvae/images/train \
        --samples "SD VAE"=./results/sit-chexpert-sdvae/samples/0400000 \
                  "MedVAE"=./results/sit-chexpert-medvae/samples/0400000 \
                  "MedVAE + REPA"=./results/sit-chexpert-medvae-repa/samples/0400000 \
                  "SD VAE + REPA"=./results/sit-chexpert-sdvae-repa/samples/0400000
"""
import argparse
import multiprocessing

import torch
from cleanfid import fid
from torch_fidelity import calculate_metrics

# clean-fid's DataLoader workers use a non-picklable local closure for image
# resizing; Python's multiprocessing default start method varies by version/OS
# (e.g. Python 3.13+ on some platforms), and anything but 'fork' fails trying
# to pickle that closure. Force 'fork' (harmless -- it's already the default
# on Linux, the actual target environment) before any multiprocessing starts.
try:
    multiprocessing.set_start_method('fork')
except RuntimeError:
    pass  # already set (e.g. re-imported) -- fine if it's already 'fork'


def parse_samples(pairs):
    samples = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f'--samples entries must be NAME=DIR, got: {pair}')
        name, path = pair.split('=', 1)
        samples[name] = path
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True,
                        help='Directory of real reference images (e.g. CheXpert train split)')
    parser.add_argument('--samples', type=str, nargs='+', required=True,
                        help='One or more NAME=DIR pairs, e.g. "SD VAE"=./results/.../samples/0400000')
    parser.add_argument('--stats-name', type=str, default='chexpert_train',
                        help='clean-fid custom-stats cache name for --ref, reused across runs/experiments')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    samples = parse_samples(args.samples)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('=== FID (torch-fidelity) ===')
    for name, gen_dir in samples.items():
        print(f'\n[{name}]')
        calculate_metrics(input1=gen_dir, input2=args.ref, cuda=use_cuda,
                          fid=True, samples_find_deep=True)

    print('\n=== Clean-FID & KID ===')
    if fid.test_stats_exists(args.stats_name, mode='clean'):
        print('Stats already exist, skipping.')
    else:
        fid.make_custom_stats(name=args.stats_name, fdir=args.ref, mode='clean',
                              batch_size=args.batch_size, device=device)

    print(f"\n{'Model':<30} {'Clean-FID':>10} {'KID':>12}")
    print('-' * 54)
    for name, gen_dir in samples.items():
        score = fid.compute_fid(fdir1=gen_dir, dataset_name=args.stats_name, dataset_res='na',
                                dataset_split='custom', mode='clean', device=device)
        kid = fid.compute_kid(fdir1=gen_dir, dataset_name=args.stats_name, dataset_res='na',
                              dataset_split='custom', mode='clean', device=device)
        print(f'{name:<30} {score:>10.4f} {kid:>12.6f}')


if __name__ == '__main__':
    main()
