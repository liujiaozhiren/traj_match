from __future__ import annotations

import torch
import torch.nn.functional as F

import cfg
from my_diffusion.base.model import MyBaseDiff
from diffusion.unet84 import AssistTrajUnet84


class FixLenDiff84(MyBaseDiff):
    """
    FixLenDiff variant using UNet84 (info_len=8, x_len=4).
    """

    def __init__(self, device=cfg.device, lr=1e-6):
        super().__init__(device=device)
        self.unet = AssistTrajUnet84().to(device)
        self.optim = torch.optim.AdamW(self.unet.parameters(), lr=lr)

    def learn(self, input_trajs, lam=None):
        info = input_trajs[..., : cfg.diff_pre_len]
        target = input_trajs[..., cfg.diff_pre_len :]
        info = info.to(self.device)
        x0 = target.to(self.device)

        t = torch.randint(low=0, high=self.n_steps, size=(len(x0) // 2 + 1,), device=self.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[: len(x0)]
        xt, noise = self.q_xt_x0(x0, t)
        pred_noise = self.unet(xt.float(), info, t)

        loss_eps = F.mse_loss(noise.float(), pred_noise)
        x0_hat = self.predict_x0(xt, t, pred_noise)
        recon_mse = F.mse_loss(x0_hat, x0)
        if lam is None:
            lam = getattr(cfg, "lambda_x0", 0.0)
        loss = loss_eps + lam * recon_mse

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), recon_mse.item()

    @torch.no_grad()
    def ddpm_step(self, xt: torch.Tensor, info: torch.Tensor, t: torch.Tensor, extra=None) -> torch.Tensor:
        eps_t = self.unet(xt.float(), info, t, extra=extra)
        alpha_t = self.gather(self.alpha, t)
        beta_t = self.gather(self.beta, t)
        sqrt_recip_alpha_t = (1.0 / (alpha_t + 1e-8)).sqrt()
        sqrt_one_minus_ab = (1.0 - self.gather(self.alpha_bar, t)).sqrt()
        mu_theta = sqrt_recip_alpha_t * (xt - (beta_t / (sqrt_one_minus_ab + 1e-8)) * eps_t)
        noise = torch.randn_like(xt) * (beta_t.sqrt())
        mask = (t > 0).float().reshape(-1, 1, 1)
        return mu_theta + mask * noise

    @torch.no_grad()
    def denoise_from(self, xt: torch.Tensor, info: torch.Tensor, t: torch.Tensor, extra=None) -> torch.Tensor:
        B = xt.size(0)
        cur = xt
        max_t = int(t.max().item())
        for step in range(max_t, -1, -1):
            step_t = torch.full((B,), step, device=xt.device, dtype=t.dtype)
            update_mask = (t >= step).float().reshape(-1, 1, 1)
            x_prev = self.ddpm_step(cur, info, step_t, extra=extra)
            cur = update_mask * x_prev + (1.0 - update_mask) * cur
        return cur

    @torch.no_grad()
    def infer_from_noise(self, info, extra=None, num=None):
        B, C, _ = info.shape
        device = info.device
        L = cfg.diff_infer_len
        last_step = self.n_steps - 1
        if num is None:
            xT = torch.randn(B, C, L, device=device)
            tT = torch.full((B,), last_step, device=device, dtype=torch.long)
            return self.denoise_from(xT, info, tT, extra=extra)
        num = int(num)
        info_rep = info.repeat_interleave(num, dim=0)
        xT = torch.randn(B * num, C, L, device=device)
        tT = torch.full((B * num,), last_step, device=device, dtype=torch.long)
        x0 = self.denoise_from(xT, info_rep, tT, extra=extra)
        return x0.reshape(B, num, C, L)

