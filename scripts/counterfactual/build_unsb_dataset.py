"""
build_unsb_dataset.py
----------------------
Bridges our chexpert_256_sdvae layout into the trainA/trainB/testA/testB
folder layout UNSB's UnalignedDataset expects.

Input (produced by prepare_chexpert.py):
    <data-dir>/images/<split>/<idx>.png
    <data-dir>/vae-sd/<split>/dataset.json   {"labels": [["<idx>.npy", class_idx], ...]}

Output:
    <out-dir>/trainA/<idx>.png   (label 0 = Healthy)
    <out-dir>/trainB/<idx>.png   (label 1 = first --pathology, e.g. Pleural Effusion)
    <out-dir>/testA/<idx>.png
    <out-dir>/testB/<idx>.png

Images are symlinked, not copied, so this is cheap to rerun / redo with a
different label split.

Usage:
    python scripts/counterfactual/build_unsb_dataset.py \
        --data-dir ./data/chexpert_256_sdvae \
        --out-dir ./data/chexpert_unsb
"""
import argparse
import json
import os


def link_split(data_dir, out_dir, split, label_a, label_b, suffix_a, suffix_b):
    images_dir = os.path.join(data_dir, 'images', split)
    features_dir = os.path.join(data_dir, 'vae-sd', split)
    with open(os.path.join(features_dir, 'dataset.json'), 'r') as f:
        entries = json.load(f)['labels']  # [["<idx>.npy", class_idx], ...]

    dir_a = os.path.join(out_dir, suffix_a)
    dir_b = os.path.join(out_dir, suffix_b)
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    n_a = n_b = 0
    for fname, label in entries:
        base = os.path.splitext(fname)[0] + '.png'
        src = os.path.abspath(os.path.join(images_dir, base))
        if not os.path.isfile(src):
            continue
        if label == label_a:
            dst = os.path.join(dir_a, base)
            n_a += 1
        elif label == label_b:
            dst = os.path.join(dir_b, base)
            n_b += 1
        else:
            continue
        if not os.path.exists(dst):
            os.symlink(src, dst)

    print(f'[{split}] {suffix_a}: {n_a} images, {suffix_b}: {n_b} images')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='chexpert_256_sdvae-style directory (has images/ and vae-sd/)')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='where to write trainA/trainB/testA/testB')
    parser.add_argument('--label-a', type=int, default=0, help='class index for domain A (default: 0 = Healthy)')
    parser.add_argument('--label-b', type=int, default=1, help='class index for domain B (default: 1 = first pathology)')
    args = parser.parse_args()

    link_split(args.data_dir, args.out_dir, 'train', args.label_a, args.label_b, 'trainA', 'trainB')
    link_split(args.data_dir, args.out_dir, 'test', args.label_a, args.label_b, 'testA', 'testB')


if __name__ == '__main__':
    main()
