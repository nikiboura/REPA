"""
Visualize counterfactual edits as a signed difference map: green = addition
(counterfactual is brighter than source, e.g. new opacity), red = removal
(counterfactual is darker than source), following the figure convention used
in "Latent Drifting in Diffusion Models for Counterfactual Medical Image
Synthesis" (Yeganeh et al.) — a standard signed pixel-difference visualization,
not a method from that paper itself.

Takes the {stem}_healthy.png / {stem}_pe.png pairs produced by
generate_counterfactual.py and saves, per pair, a side-by-side panel:
source | counterfactual | diff (overlaid on source for anatomical context).
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image
import wandb


def compute_diff_overlay(source_gray, target_gray, alpha=0.6, threshold=0.0):
    """
    source_gray, target_gray: (H, W) uint8 arrays.
    Returns an (H, W, 3) uint8 RGB image: grayscale source with a green/red
    overlay showing where the target added (green) or removed (red) intensity.

    threshold: fraction (0-1) of the max abs diff below which pixels are
    treated as noise and left uncolored (denoises near-uniform VAE/sampling
    jitter that would otherwise tint the whole image).
    """
    diff = target_gray.astype(np.float32) - source_gray.astype(np.float32)  # [-255, 255]

    max_abs = np.abs(diff).max()
    if max_abs < 1e-6:
        norm_diff = np.zeros_like(diff)
    else:
        norm_diff = diff / max_abs  # [-1, 1]

    if threshold > 0:
        norm_diff = np.where(np.abs(norm_diff) < threshold, 0.0, norm_diff)

    addition = np.clip(norm_diff, 0, None)   # target brighter than source
    removal = np.clip(-norm_diff, 0, None)   # target darker than source

    base_rgb = np.stack([source_gray] * 3, axis=-1).astype(np.float32)

    overlay = base_rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + removal * 255. * alpha, 0, 255)   # red channel
    overlay[..., 1] = np.clip(overlay[..., 1] + addition * 255. * alpha, 0, 255)  # green channel
    overlay[..., 2] = np.clip(overlay[..., 2] * (1 - np.maximum(addition, removal) * alpha), 0, 255)

    return overlay.astype(np.uint8)


def make_panel(source_rgb, target_rgb, diff_rgb):
    """Concatenate source | target | diff horizontally into one image."""
    h = source_rgb.shape[0]
    imgs = [Image.fromarray(a) for a in (source_rgb, target_rgb, diff_rgb)]
    widths = [im.width for im in imgs]
    panel = Image.new('RGB', (sum(widths), h), color=(0, 0, 0))
    x = 0
    for im in imgs:
        panel.paste(im, (x, 0))
        x += im.width
    return panel


def process_pair(healthy_path, pe_path, out_path, alpha, threshold):
    healthy = Image.open(healthy_path).convert('RGB')
    pe = Image.open(pe_path).convert('RGB')
    if healthy.size != pe.size:
        pe = pe.resize(healthy.size)

    healthy_gray = np.array(healthy.convert('L'))
    pe_gray = np.array(pe.convert('L'))

    diff_rgb = compute_diff_overlay(healthy_gray, pe_gray, alpha=alpha, threshold=threshold)
    panel = make_panel(np.array(healthy), np.array(pe), diff_rgb)
    panel.save(out_path)

    return np.array(healthy), np.array(pe), diff_rgb


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    healthy_files = sorted(glob.glob(os.path.join(args.pairs_dir, '*_healthy.png')))
    if not healthy_files:
        print(f'No *_healthy.png files found in {args.pairs_dir}')
        return

    if args.report_to == 'wandb':
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        wandb_table = wandb.Table(columns=['id', 'healthy', 'counterfactual_pe', 'diff'])

    count = 0
    for healthy_path in healthy_files:
        stem = os.path.basename(healthy_path).replace('_healthy.png', '')
        pe_path = os.path.join(args.pairs_dir, f'{stem}_pe.png')
        if not os.path.exists(pe_path):
            print(f'Skipping {stem}: no matching _pe.png')
            continue

        out_path = os.path.join(args.output_dir, f'{stem}_diff.png')
        healthy_arr, pe_arr, diff_arr = process_pair(healthy_path, pe_path, out_path, args.alpha, args.threshold)
        if args.report_to == 'wandb':
            wandb_table.add_data(stem, wandb.Image(healthy_arr), wandb.Image(pe_arr), wandb.Image(diff_arr))
        count += 1

    if args.report_to == 'wandb':
        wandb.log({'diff_visualizations': wandb_table})
        wandb.finish()

    print(f'Saved {count} diff panels to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs-dir', type=str, required=True,
                        help="Directory containing {stem}_healthy.png / {stem}_pe.png pairs "
                             "(the output of generate_counterfactual.py).")
    parser.add_argument('--output-dir', type=str, default='./diff_visualizations')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help="Overlay strength for the green/red diff coloring (0-1).")
    parser.add_argument('--threshold', type=float, default=0.05,
                        help="Fraction of max abs diff below which pixels are left uncolored, "
                             "to suppress near-uniform sampling noise.")

    parser.add_argument('--report-to', type=str, default='none', choices=['none', 'wandb'],
                        help="Log healthy/pe/diff triples to Weights & Biases as a table.")
    parser.add_argument('--wandb-project', type=str, default='REPA')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help="W&B run name; defaults to wandb's auto-generated name if unset.")
    args = parser.parse_args()

    main(args)
