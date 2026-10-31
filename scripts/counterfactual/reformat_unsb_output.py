"""
reformat_unsb_output.py
------------------------
UNSB's test.py saves, per source image `<idx>.png`:
    results/<name>/test_<phase>_<epoch>/images/real/<idx>.png       (source, Healthy)
    results/<name>/test_<phase>_<epoch>/images/fake_<t>/<idx>.png   (translated, after NFE=t steps)

Note: at test time sb_model.py names the source-image visual 'real', not
'real_A' -- that's only the training-time name (models/sb_model.py __init__:
`self.visual_names = ['real_A', ...]` during train, but reassigned to
`['real', 'fake_1', ...]` when `self.opt.phase == 'test'`).

Our compute_metrics.py (used for the REPA/I2SB counterfactual pipeline) expects
a single flat directory with paired filenames:
    <idx>_healthy.png
    <idx>_pe.png

This script symlinks UNSB's output into that shape so the same
compute_metrics.py can score UNSB as a baseline, apples-to-apples.

Usage:
    python scripts/counterfactual/reformat_unsb_output.py \
        --unsb-results-dir ./third_party/UNSB/results/chexpert_h2pe_SB/test_latest/images \
        --out-dir ./results/unsb-chexpert-baseline/counterfactuals \
        --nfe 5
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unsb-results-dir', type=str, required=True,
                        help="UNSB's .../test_<phase>_<epoch>/images directory (contains real/, fake_1/, ...)")
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--nfe', type=int, default=None,
                        help='which fake_<nfe> folder to use as the final translation (default: highest available)')
    args = parser.parse_args()

    real_a_dir = os.path.join(args.unsb_results_dir, 'real')
    if not os.path.isdir(real_a_dir):
        raise FileNotFoundError(f'{real_a_dir} not found -- did test.py run successfully?')

    if args.nfe is None:
        nfes = []
        for name in os.listdir(args.unsb_results_dir):
            if name.startswith('fake_'):
                try:
                    nfes.append(int(name[len('fake_'):]))
                except ValueError:
                    pass
        if not nfes:
            raise FileNotFoundError(f'no fake_<N> folders found in {args.unsb_results_dir}')
        args.nfe = max(nfes)
    fake_dir = os.path.join(args.unsb_results_dir, f'fake_{args.nfe}')
    if not os.path.isdir(fake_dir):
        raise FileNotFoundError(f'{fake_dir} not found')

    print(f'Using fake_{args.nfe} as the final translation.')
    os.makedirs(args.out_dir, exist_ok=True)

    n = 0
    for fname in sorted(os.listdir(fake_dir)):
        stem = os.path.splitext(fname)[0]
        src_healthy = os.path.join(real_a_dir, fname)
        src_pe = os.path.join(fake_dir, fname)
        if not os.path.isfile(src_healthy):
            continue
        dst_healthy = os.path.join(args.out_dir, f'{stem}_healthy.png')
        dst_pe = os.path.join(args.out_dir, f'{stem}_pe.png')
        if not os.path.exists(dst_healthy):
            os.symlink(os.path.abspath(src_healthy), dst_healthy)
        if not os.path.exists(dst_pe):
            os.symlink(os.path.abspath(src_pe), dst_pe)
        n += 1

    print(f'Wrote {n} healthy/pe pairs to {args.out_dir}')


if __name__ == '__main__':
    main()
