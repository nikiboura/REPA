import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[],
            accelerator=None,
            latents_scale=None,
            latents_bias=None,
            beta_max=0.3,
            cond_x1=False,
            ot_ode=False,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
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

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None, x_source=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        if x_source is not None:
            # I2SB Proposition 3.3 (Eq. 11): q(Xt|X0, X1) = N(Xt; µt, Σt)
            #   X0 = images  (target: PE,      t=0)
            #   X1 = x_source (source: Healthy, t=1)
            sigma_sq_t     = self._sigma_sq(time_input)
            sigma_bar_sq_t = self._sigma_bar_sq(time_input)
            sigma_sq_total = sigma_sq_t + sigma_bar_sq_t  # = beta_max*2/3 (constant)

            # µt: weighted mean between X0 and X1
            mu_t = (sigma_bar_sq_t / sigma_sq_total) * images + \
                   (sigma_sq_t     / sigma_sq_total) * x_source

            # Σt: bridge variance (zero at both endpoints, max at t=0.5)
            variance_t = (sigma_sq_t * sigma_bar_sq_t / sigma_sq_total).clamp(min=1e-8)

            # Sample Xt ~ q(Xt|X0, X1)  (ot_ode: deterministic, no bridge noise — matches I2SB's ot_ode flag)
            if self.ot_ode:
                model_input = mu_t
            else:
                eps = torch.randn_like(images)
                model_input = mu_t + variance_t.sqrt() * eps

            # Eq. 12: model predicts (Xt - X0) / σt
            sigma_t = sigma_sq_t.sqrt().clamp(min=1e-4)
            model_target = (model_input - images) / sigma_t
        else:
            noises = torch.randn_like(images)
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            model_input = alpha_t * images + sigma_t * noises
            if self.prediction == 'v':
                model_target = d_alpha_t * images + d_sigma_t * noises
            else:
                raise NotImplementedError()

        cond = x_source if (self.cond_x1 and x_source is not None) else None
        model_output, zs_tilde  = model(model_input, time_input.flatten(), cond=cond, **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss
        proj_loss = 0.
        if len(zs) == 0:
            return denoising_loss, torch.tensor(0., device=denoising_loss.device)
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            if z.shape[1] != z_tilde.shape[1]:
                B, N, D = z.shape
                H = W = int(N ** 0.5)
                H2 = W2 = int(z_tilde.shape[1] ** 0.5)
                z = z.reshape(B, H, W, D).permute(0, 3, 1, 2)
                z = F.interpolate(z.float(), size=(H2, W2), mode="bilinear", align_corners=False)
                z = z.permute(0, 2, 3, 1).reshape(B, H2 * W2, D)
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        return denoising_loss, proj_loss
