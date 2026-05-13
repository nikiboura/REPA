"""
train.py  

automatically computes encoder-depth as 28% of model depth (matching the
original REPA) unless --encoder-depth can be used.

Usage:
    #REPA
    python scripts/train.py \
        --model      SiT-S/8 \
        --exp-name   sit-s8-repa-sd \
        --data-dir   /data/chexpert_256 \
        --vae-type   sd \
        --enc-type   meddinov3-vit-b \
        --proj-coeff 0.5 \
        --output-dir /results \
        [--encoder-depth 3]          # override auto-compute
        
    # No REPA:
    python scripts/train.py \
        --model      SiT-S/8 \
        --exp-name   sit-s8-baseline \
        --data-dir   /data/chexpert_256 \
        --vae-type   sd \
        --enc-type   None \
        --proj-coeff 0.0 \
        --output-dir /results
"""

import argparse
import math
import os
import subprocess
import sys

REPA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Total transformer depth per model family
MODEL_DEPTHS = {
    "SiT-XL": 28,
    "SiT-L":  24,
    "SiT-B":  12,
    "SiT-S":  12,
}

# Default projector output dim per encoder
ENCODER_DIMS = {
    "dinov2-vit-b":    768,
    "dinov2-vit-l":    1024,
    "meddinov3-vit-b": 768,
    "clip-vit-b":      512,
    "mae-vit-b":       768,
    "jepa-vit-b":      768,
    "None":            768,
}


def get_model_depth(model_name):
    family = model_name.split("/")[0]  # e.g. "SiT-S"
    if family not in MODEL_DEPTHS:
        raise ValueError(f"Unknown model family '{family}'. Known: {list(MODEL_DEPTHS)}")
    return MODEL_DEPTHS[family]


def auto_encoder_depth(model_name, pct):
    depth = get_model_depth(model_name)
    return max(1, round(depth * pct))


def main():
    parser = argparse.ArgumentParser(description="REPA training launcher.")

    #core
    parser.add_argument("--model",       required=True, help="e.g. SiT-S/8, SiT-B/4, SiT-XL/2")
    parser.add_argument("--exp-name",    required=True)
    parser.add_argument("--data-dir",    required=True, help="Path to prepared chexpert_256/")
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--vae-type",    default="sd", choices=["sd", "medvae"])

    # REPA
    parser.add_argument("--enc-type",    default="meddinov3-vit-b",
                        help="Encoder type or 'None' to disable REPA")
    parser.add_argument("--proj-coeff",  type=float, default=0.5)
    parser.add_argument("--encoder-depth", type=int, default=None,
                        help="REPA alignment layer. Auto-computed if not given.")
    parser.add_argument("--encoder-depth-pct", type=float, default=0.28,
                        help="Fraction of model depth for REPA (default 0.28, matching original paper)")

    # training
    parser.add_argument("--num-classes",   type=int,   default=2)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=400000)
    parser.add_argument("--cfg-prob",      type=float, default=0.1)
    parser.add_argument("--num-workers",   type=int,   default=4)
    parser.add_argument("--resolution",    type=int,   default=256)
    parser.add_argument("--path-type",     default="linear")
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--num-processes", type=int,   default=1)
    parser.add_argument("--teacher-ckpt",  default=None)
    parser.add_argument("--report-to",     default="none")

    args = parser.parse_args()

    # auto encoder-depth 
    if args.encoder_depth is not None:
        enc_depth = args.encoder_depth
        print(f"encoder-depth: {enc_depth} (explicit)")
    else:
        enc_depth = auto_encoder_depth(args.model, args.encoder_depth_pct)
        total = get_model_depth(args.model)
        print(f"encoder-depth: {enc_depth} / {total} "
              f"({args.encoder_depth_pct*100:.0f}% of model depth, auto-computed)")

    # projector-embed-dims
    proj_dim = ENCODER_DIMS.get(args.enc_type, 768)

    cmd = [
        "accelerate", "launch",
        "--mixed_precision", args.mixed_precision,
        "--num_processes",   str(args.num_processes),
        os.path.join(REPA_DIR, "train.py"),
        "--exp-name",              args.exp_name,
        "--output-dir",            args.output_dir,
        "--report-to",             args.report_to,
        "--model",                 args.model,
        "--num-classes",           str(args.num_classes),
        "--encoder-depth",         str(enc_depth),
        "--enc-type",              args.enc_type,
        "--proj-coeff",            str(args.proj_coeff),
        "--projector-embed-dims",  str(proj_dim),
        "--data-dir",              args.data_dir,
        "--resolution",            str(args.resolution),
        "--batch-size",            str(args.batch_size),
        "--num-workers",           str(args.num_workers),
        "--learning-rate",         str(args.learning_rate),
        "--mixed-precision",       args.mixed_precision,
        "--cfg-prob",              str(args.cfg_prob),
        "--path-type",             args.path_type,
        "--max-train-steps",       str(args.max_train_steps),
        "--checkpointing-steps",   str(args.checkpointing_steps),
        "--sampling-steps",        "99999999",
        "--vae-type",              args.vae_type,
    ]

    if args.teacher_ckpt:
        cmd += ["--teacher-ckpt", args.teacher_ckpt]

    print("\nRunning:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, cwd=REPA_DIR, check=True)


if __name__ == "__main__":
    main()
