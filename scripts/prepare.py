"""
prepare.py  —  Download CheXpert and encode VAE latents using either sd or medvae.

Usage:
    python scripts/prepare.py \
        --data-root /data/chexpert \         # raw dataset saved here
        --out-dir   /data/chexpert_256 \     # processed dataset saved here
        --vae-type  sd                       # sd | medvae | both 
        --max-samples 80000 \
        --resolution  256 \
        --kaggle-json ~/.kaggle/kaggle.json

Output:
    <out-dir>/images/train/        PNG images  (shared, used for FID)
    <out-dir>/vae-sd/train/        SD VAE latents
    <out-dir>/vae-medvae/train/    MedVAE latents
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


REPA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP_SCRIPT = os.path.join(REPA_DIR, "dataset_preparation_scripts", "prepare_chexpert.py")


def download_chexpert(data_root, kaggle_json):
    if os.path.isfile(os.path.join(data_root, "train.csv")):
        print("CheXpert already downloaded.")
        return

    if kaggle_json and os.path.isfile(kaggle_json):
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        dest = os.path.join(kaggle_dir, "kaggle.json")
        shutil.copy(kaggle_json, dest)
        os.chmod(dest, 0o600)

    print("Downloading CheXpert from Kaggle")
    subprocess.run(
        ["kaggle", "datasets", "download", "ashery/chexpert",
         "-p", data_root, "--unzip"],
        check=True,
    )
    print("Download complete.")


def encode(data_root, out_dir, vae_type, resolution, max_samples):
    if vae_type == "medvae":
        # encode into temp dir, then move latents to vae-medvae/
        tmp = tempfile.mkdtemp()
        try:
            subprocess.run(
                [sys.executable, PREP_SCRIPT,
                 "--chexpert-root", data_root,
                 "--out-dir",       tmp,
                 "--resolution",    str(resolution),
                 "--vae-type",      "medvae",
                 "--max-samples",   str(max_samples)],
                check=True,
            )
            dst = os.path.join(out_dir, "vae-medvae")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(os.path.join(tmp, "vae-sd"), dst)
            # copy images if not already there (from SD run)
            img_dst = os.path.join(out_dir, "images")
            img_src = os.path.join(tmp, "images")
            if not os.path.exists(img_dst) and os.path.exists(img_src):
                shutil.move(img_src, img_dst)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        subprocess.run(
            [sys.executable, PREP_SCRIPT,
             "--chexpert-root", data_root,
             "--out-dir",       out_dir,
             "--resolution",    str(resolution),
             "--vae-type",      "sd",
             "--max-samples",   str(max_samples)],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare CheXpert dataset.")
    parser.add_argument("--data-root",   required=True, help="Where to download raw CheXpert")
    parser.add_argument("--out-dir",     required=True, help="Where to save processed data")
    parser.add_argument("--vae-type",    default="both", choices=["sd", "medvae", "both"])
    parser.add_argument("--max-samples", type=int, default=80000)
    parser.add_argument("--resolution",  type=int, default=256)
    parser.add_argument("--kaggle-json", default=None, help="Path to kaggle.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    download_chexpert(args.data_root, args.kaggle_json)

    vae_types = ["sd", "medvae"] if args.vae_type == "both" else [args.vae_type]
    for vt in vae_types:
        out_key = "vae-sd" if vt == "sd" else "vae-medvae"
        if os.path.isfile(os.path.join(args.out_dir, out_key, "train", "dataset.json")):
            print(f"{vt} latents already exist, skipping.")
            continue
        print(f"\nEncoding with {vt} VAE...")
        encode(args.data_root, args.out_dir, vt, args.resolution, args.max_samples)

    print("\nDone. Layout:")
    print(f"  Images : {args.out_dir}/images/")
    print(f"  SD VAE : {args.out_dir}/vae-sd/")
    print(f"  MedVAE : {args.out_dir}/vae-medvae/")


if __name__ == "__main__":
    main()
