import torch
import os
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from dataset import LatentDataset


@torch.no_grad()
def sample_posterior(moments, latents_scale, latents_bias):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    return z * latents_scale + latents_bias


def _sigma_sq(t, beta_max):
    """σ²t = ∫₀ᵗ βτ dτ  where βt = 4*beta_max*t*(1-t)"""
    return beta_max * (2 * t ** 2 - 4 * t ** 3 / 3)


@torch.no_grad()
def ddpm_bridge_sampler(model, z_source, y, num_steps, beta_max, cfg_scale, cond=None, ot_ode=False):
    """
    I2SB Algorithm 2: DDPM backward from X1 (Healthy, t=1) to X0 (PE, t=0).

    At each step n:
      1. Predict ε = (Xt - X0) / σt  →  recover X0_pred = Xt - σt * ε_pred
      2. Sample X_{n-1} ~ p(X_{n-1} | X0_pred, Xn)  using the DDPM posterior
         derived in Proof of Proposition 3.3.

    cond: optional source-latent conditioning (I2SB's cond_x1) — constant across all steps,
          must match whatever `--cond-x1` setting the checkpoint was trained with.
    ot_ode: deterministic bridge (no noise in the posterior step) — matches official I2SB's
            --ot-ode; must match training.
    """
    device = z_source.device
    dtype  = z_source.dtype

    # Discrete time grid: t goes from 1 → 0 in num_steps steps
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)  # shape (num_steps+1,)

    sigma_sq_total = beta_max * 2 / 3   # = ∫₀¹ βτ dτ  (constant)

    if cfg_scale > 1.0:
        y_null = torch.tensor([model.num_classes] * y.shape[0], device=device)

    Xn = z_source.clone().double()

    for i in range(num_steps):
        t_cur  = ts[i]      # current time  (high → low)
        t_prev = ts[i + 1]  # previous time

        # ── 1. model forward (with optional CFG) ──────────────────────────────
        if cfg_scale > 1.0:
            m_in  = torch.cat([Xn.to(dtype)] * 2)
            y_in  = torch.cat([y, y_null])
            cond_in = torch.cat([cond, cond]) if cond is not None else None
        else:
            m_in = Xn.to(dtype)
            y_in = y
            cond_in = cond

        # t_input batch size must track m_in (doubled under CFG), not Xn
        t_input = torch.full((m_in.shape[0],), t_cur.item(), device=device, dtype=dtype)

        eps_pred = model(m_in, t_input, y=y_in, cond=cond_in)[0].double()

        if cfg_scale > 1.0:
            eps_cond, eps_uncond = eps_pred.chunk(2)
            eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        # ── 2. recover X0  (from Eq. 12 inversion: X0 = Xt - σt * ε_pred) ──
        ssq_cur = _sigma_sq(t_cur, beta_max)
        sig_cur = ssq_cur.sqrt().clamp(min=1e-4)
        X0_pred = Xn - sig_cur * eps_pred

        # ── 3. DDPM posterior step  (Proof of Proposition 3.3) ───────────────
        if i == num_steps - 1:
            # Last step: return X0 directly (no noise injected at t=0)
            Xn = X0_pred
        else:
            ssq_prev  = _sigma_sq(t_prev, beta_max)
            alpha_sq  = (ssq_cur - ssq_prev).clamp(min=1e-8)   # incremental variance
            denom     = alpha_sq + ssq_prev

            # Posterior mean
            mu_prev = (alpha_sq / denom) * X0_pred + (ssq_prev / denom) * Xn

            if ot_ode:
                Xn = mu_prev
            else:
                # Posterior variance  (zero when ssq_prev=0, i.e. at t→0)
                var_prev = (alpha_sq * ssq_prev / denom).clamp(min=0.0)
                Xn = mu_prev + var_prev.sqrt() * torch.randn_like(Xn)

    return Xn.to(dtype)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    latent_size = args.resolution // 8
    z_dims = [int(z) for z in args.projector_embed_dims.split(',') if z] if args.projector_embed_dims else []

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    cond_channels = 4 if args.cond_x1 else 0
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_scale > 1.0),
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        cond_channels=cond_channels,
        **block_kwargs,
    ).to(device)

    state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)['ema']
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{args.vae}').to(device)
    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias  = torch.zeros(1, 4, 1, 1).to(device)

    # Only Healthy images (label=0) as source — they are X1 in the bridge
    dataset = LatentDataset(args.data_dir, split=args.split)
    dataset.entries = [e for e in dataset.entries if e[1] == 0]
    print(f'Found {len(dataset)} Healthy images to translate → PE')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    for raw_images, moments, _ in dataloader:
        moments = moments.to(device)

        # X1 = Healthy latent (starting point of the bridge, t=1)
        z_source = sample_posterior(moments, latents_scale, latents_bias)

        # Target label: PE (label=1)
        y_pe = torch.ones(z_source.shape[0], dtype=torch.long, device=device)

        # Algorithm 2: DDPM backward  X1 (Healthy) → X0 (PE)
        cond = z_source if args.cond_x1 else None
        z_pe = ddpm_bridge_sampler(
            model=model,
            z_source=z_source,
            y=y_pe,
            num_steps=args.num_steps,
            beta_max=args.beta_max,
            cfg_scale=args.cfg_scale,
            cond=cond,
            ot_ode=args.ot_ode,
        ).to(torch.float32)

        # Decode generated PE latent → pixel image
        decoded = vae.decode((z_pe - latents_bias) / latents_scale)
        samples_pe = decoded if isinstance(decoded, torch.Tensor) else decoded.sample
        samples_pe = torch.clamp(255. * (samples_pe + 1) / 2., 0, 255).permute(0, 2, 3, 1).to('cpu', dtype=torch.uint8).numpy()

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

    parser.add_argument('--ckpt',    type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./generated_pe')
    parser.add_argument('--split',   type=str, default='test')

    parser.add_argument('--model',   type=str, default='SiT-S/4')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--vae',     type=str, default='mse', choices=['ema', 'mse'])
    parser.add_argument('--projector-embed-dims', type=str, default='')
    parser.add_argument('--fused-attn', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--qk-norm',    action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--encoder-depth', type=int, default=None)

    parser.add_argument('--num-steps', type=int, default=200,
                        help='DDPM backward steps (paper uses 1000; 200 is faster with little quality loss)')
    parser.add_argument('--beta-max', type=float, default=0.3,
                        help='Controls bridge width; must match value used in training')
    parser.add_argument('--cond-x1', action=argparse.BooleanOptionalAction, default=False,
                        help='Condition the network on the source (Healthy) latent; must match training.')
    parser.add_argument('--ot-ode', action=argparse.BooleanOptionalAction, default=False,
                        help='Deterministic bridge sampling (no posterior noise); must match training.')
    parser.add_argument('--cfg-scale', type=float, default=3.0)

    parser.add_argument('--batch-size',  type=int, default=8)
    parser.add_argument('--num-samples', type=int, default=-1)

    args = parser.parse_args()

    if args.encoder_depth is None:
        MODEL_DEPTHS = {'SiT-XL': 28, 'SiT-L': 24, 'SiT-B': 12, 'SiT-S': 12}
        family = args.model.split('/')[0]
        total_depth = MODEL_DEPTHS.get(family, 12)
        args.encoder_depth = max(1, round(total_depth * 0.28))

    main(args)
