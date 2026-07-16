import torch
import os
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from samplers import euler_sampler
from dataset import LatentDataset


@torch.no_grad()
def sample_posterior(moments, latents_scale, latents_bias):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    return z * latents_scale + latents_bias


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    latent_size = args.resolution // 8
    z_dims = [int(z) for z in args.projector_embed_dims.split(',') if z] if args.projector_embed_dims else []

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_scale > 1.0),
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)

    state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)['ema']
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{args.vae}').to(device)
    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.zeros(1, 4, 1, 1).to(device)

    # Keep only Healthy images (label=0) — they are the source (t=1 starting point)
    dataset = LatentDataset(args.data_dir, split=args.split)
    dataset.entries = [e for e in dataset.entries if e[1] == 0]
    print(f'Found {len(dataset)} Healthy images to translate → PE')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    for raw_images, moments, _ in dataloader:
        moments = moments.to(device)

        # z_source: latent of the real Healthy image — this is t=1 in the bridge
        z_source = sample_posterior(moments, latents_scale, latents_bias)

        # Target label: PE (label=1)
        y_pe = torch.ones(z_source.shape[0], dtype=torch.long, device=device)

        # Run the full bridge from t=1 (Healthy image) to t=0 (PE image)
        # The model learned: x_t = (1-t)*x_PE + t*x_Healthy
        # so starting from x_Healthy at t=1 and integrating to t=0 gives x_PE
        z_pe = euler_sampler(
            model=model,
            latents=z_source,
            y=y_pe,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            t0=1.0,
        ).to(torch.float32)

        # Decode generated PE latent → pixel image
        decoded = vae.decode((z_pe - latents_bias) / latents_scale)
        samples_pe = decoded if isinstance(decoded, torch.Tensor) else decoded.sample
        samples_pe = torch.clamp(255. * (samples_pe + 1) / 2., 0, 255).permute(0, 2, 3, 1).to('cpu', dtype=torch.uint8).numpy()

        # Real Healthy images come directly from the dataset loader (no VAE round-trip)
        samples_real = raw_images.permute(0, 2, 3, 1).cpu().numpy()

        for i, (real, pe) in enumerate(zip(samples_real, samples_pe)):
            stem = f'{total + i:06d}'
            Image.fromarray(real).save(os.path.join(args.output_dir, f'{stem}_healthy.png'))
            Image.fromarray(pe).save(os.path.join(args.output_dir, f'{stem}_pe.png'))

        total += len(samples_pe)
        print(f'Generated {total} pairs')

        if 0 < args.num_samples <= total:
            break

    print(f'Done. Saved {total} pairs to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./generated_pe')
    parser.add_argument('--split', type=str, default='test')

    parser.add_argument('--model', type=str, default='SiT-S/4')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--vae', type=str, default='mse', choices=['ema', 'mse'])
    parser.add_argument('--projector-embed-dims', type=str, default='')
    parser.add_argument('--fused-attn', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--qk-norm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--encoder-depth', type=int, default=None)

    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--cfg-scale', type=float, default=3.0)
    parser.add_argument('--guidance-low', type=float, default=0.)
    parser.add_argument('--guidance-high', type=float, default=1.)
    parser.add_argument('--path-type', type=str, default='linear')

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-samples', type=int, default=-1, help='-1 to process all Healthy images')

    args = parser.parse_args()

    if args.encoder_depth is None:
        MODEL_DEPTHS = {'SiT-XL': 28, 'SiT-L': 24, 'SiT-B': 12, 'SiT-S': 12}
        family = args.model.split('/')[0]
        total_depth = MODEL_DEPTHS.get(family, 12)
        args.encoder_depth = max(1, round(total_depth * 0.28))

    main(args)
