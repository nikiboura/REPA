"""
I2SB bridge loss — extends the standard SILoss (scripts/loss.py) with
Healthy<->PE bridge training (I2SB Proposition 3.3) and optional cond_x1/ot_ode.

When called without x_source, falls back to the standard (parent) loss
unchanged, so this class is a strict superset of the standard behavior.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F

from loss import SILoss, mean_flat


class I2SBLoss(SILoss):
    def __init__(self, *args, beta_max=0.3, cond_x1=False, ot_ode=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_max = beta_max
        self.cond_x1 = cond_x1
        self.ot_ode = ot_ode

    def _sigma_sq(self, t):
        """σ²t = ∫₀ᵗ βτ dτ  where βt = 4*beta_max*t*(1-t)  (symmetric schedule)"""
        return self.beta_max * (2 * t ** 2 - 4 * t ** 3 / 3)

    def _sigma_bar_sq(self, t):
        """σ̄²t = σ²₁ - σ²t  (variance remaining from t to 1)"""
        sigma_sq_total = self.beta_max * 2 / 3   # = ∫₀¹ βτ dτ
        return sigma_sq_total - self._sigma_sq(t)

    def __call__(self, model, images, model_kwargs=None, zs=None, x_source=None):
        if model_kwargs is None:
            model_kwargs = {}

        if x_source is None:
            # No bridge pair given: behave exactly like the standard loss.
            return super().__call__(model, images, model_kwargs=model_kwargs, zs=zs)

        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            rnd_normal = torch.randn((images.shape[0], 1, 1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / torch.pi * torch.atan(sigma)
        time_input = time_input.to(device=images.device, dtype=images.dtype)

        # I2SB Proposition 3.3 (Eq. 11): q(Xt|X0, X1) = N(Xt; µt, Σt)
        #   X0 = images   (target: PE,      t=0)
        #   X1 = x_source (source: Healthy, t=1)
        sigma_sq_t     = self._sigma_sq(time_input)
        sigma_bar_sq_t = self._sigma_bar_sq(time_input)
        sigma_sq_total = sigma_sq_t + sigma_bar_sq_t  # = beta_max*2/3 (constant)

        mu_t = (sigma_bar_sq_t / sigma_sq_total) * images + \
               (sigma_sq_t     / sigma_sq_total) * x_source
        variance_t = (sigma_sq_t * sigma_bar_sq_t / sigma_sq_total).clamp(min=1e-8)

        # ot_ode: deterministic, no bridge noise — matches I2SB's ot_ode flag
        if self.ot_ode:
            model_input = mu_t
        else:
            eps = torch.randn_like(images)
            model_input = mu_t + variance_t.sqrt() * eps

        # Eq. 12: model predicts (Xt - X0) / σt
        sigma_t = sigma_sq_t.sqrt().clamp(min=1e-4)
        model_target = (model_input - images) / sigma_t

        cond = x_source if self.cond_x1 else None
        model_output, zs_tilde = model(model_input, time_input.flatten(), cond=cond, **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        proj_loss = 0.
        if len(zs) == 0:
            return denoising_loss, torch.tensor(0., device=denoising_loss.device)
        bsz = zs[0].shape[0]
        for z, z_tilde in zip(zs, zs_tilde):
            if z.shape[1] != z_tilde.shape[1]:
                B, N, D = z.shape
                H = W = int(N ** 0.5)
                H2 = W2 = int(z_tilde.shape[1] ** 0.5)
                z = z.reshape(B, H, W, D).permute(0, 3, 1, 2)
                z = F.interpolate(z.float(), size=(H2, W2), mode="bilinear", align_corners=False)
                z = z.permute(0, 2, 3, 1).reshape(B, H2 * W2, D)
            for z_j, z_tilde_j in zip(z, z_tilde):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1)
                z_j = torch.nn.functional.normalize(z_j, dim=-1)
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        return denoising_loss, proj_loss
