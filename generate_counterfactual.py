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

# output: z0 of shape (B, 4, 32, 32) the actual latent representing the real image
@torch.no_grad()
def sample_posterior(moments, latents_scale, latents_bias):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    return z * latents_scale + latents_bias

# output: z_t0 — shape (B, 4, 32, 32), a partially noised version of the real latent
def partial_noise(x0, t0):
    noise = torch.randn_like(x0)
    return (1 - t0) * x0 + t0 * noise


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    
    latent_size = args.resolution // 8
    z_dims = [int(z) for z in args.projector_embed_dims.split(',') if z] if args.projector_embed_dims else []
    
    # load model
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_scale > 1.0),
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)

    # loads the EMA weights from the checkpoint
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)['ema']
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # load VAE for decoding image back to pixel space
    if args.vae == 'medvae':
        from medvae import MVAE
        vae = MVAE(model_name='medvae_8_4_2d', modality='xray').model.to(device).eval()
        latents_scale = torch.ones(1, 4, 1, 1).to(device)
        latents_bias = torch.zeros(1, 4, 1, 1).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{args.vae}').to(device)
        latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
        latents_bias = torch.zeros(1, 4, 1, 1).to(device)

    # load dataset
    dataset = LatentDataset(args.data_dir, split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    for raw_images, moments, labels in dataloader:
        moments = moments.to(device)
        labels = labels.to(device)

        # encode real image to clean latent
        z0 = sample_posterior(moments, latents_scale, latents_bias)

        # add noise up to t0
        z_t0 = partial_noise(z0, args.t0)

        # flip label: 0 -> 1, 1 -> 0
        y_cf = (args.num_classes - 1) - labels

        # denoise from t0 conditioned on counterfactual label
        z_cf = euler_sampler(
            model=model,
            latents=z_t0,                     # start from noised real image
            y=y_cf,                           # condition on flipped label
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low, 
            guidance_high=args.guidance_high, # CFG is applied at every denoising step
            path_type=args.path_type,
            t0=args.t0,                       # start staircase from t0, not from 1
        ).to(torch.float32)

        # decode to pixels
        decoded = vae.decode((z_cf - latents_bias) / latents_scale) # reverses the normalization
        samples = decoded if isinstance(decoded, torch.Tensor) else decoded.sample
        samples = (samples + 1) / 2.
        samples = torch.clamp(255. * samples, 0, 255).permute(0, 2, 3, 1).to('cpu', dtype=torch.uint8).numpy()

        #saves images
        for i, sample in enumerate(samples):
            original_label = labels[i].item()
            cf_label = y_cf[i].item()
            fname = f'{total + i:06d}_from{original_label}_to{cf_label}.png'
            Image.fromarray(sample).save(os.path.join(args.output_dir, fname))

        total += len(samples)
        print(f'Generated {total} counterfactuals')

        if 0 < args.num_samples <= total:
            break

    print(f'Done. Saved to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./counterfactuals')
    parser.add_argument('--split', type=str, default='test')

    parser.add_argument('--model', type=str, default='SiT-S/4')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--vae', type=str, default='mse', choices=['ema', 'mse', 'medvae'])
    parser.add_argument('--projector-embed-dims', type=str, default='')
    parser.add_argument('--fused-attn', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--qk-norm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--encoder-depth', type=int, default=None)

    parser.add_argument('--t0', type=float, default=0.5)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--cfg-scale', type=float, default=1.5)
    parser.add_argument('--guidance-low', type=float, default=0.)
    parser.add_argument('--guidance-high', type=float, default=1.)
    parser.add_argument('--path-type', type=str, default='linear')

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-samples', type=int, default=-1, help='-1 to process all images')

    args = parser.parse_args()

    if args.encoder_depth is None:
        MODEL_DEPTHS = {'SiT-XL': 28, 'SiT-L': 24, 'SiT-B': 12, 'SiT-S': 12}
        family = args.model.split('/')[0]
        total_depth = MODEL_DEPTHS.get(family, 12)
        args.encoder_depth = max(1, round(total_depth * 0.28))

    main(args)
