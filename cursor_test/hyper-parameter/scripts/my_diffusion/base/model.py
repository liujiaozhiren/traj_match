"""Minimal DDPM base (no pretrain/EMA deps). FixLenDiff84 replaces self.unet after super().__init__."""
from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg


class MyBaseDiff(nn.Module):
    def __init__(self, device=cfg.device):
        super().__init__()
        self.device = device
        self.lr = cfg.diff_lr
        self.n_steps = cfg.diffusion_timestamp
        self.beta = torch.linspace(cfg.beta_start, cfg.beta_end, self.n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.unet = nn.Identity()
        self.optim = None  # concrete subclasses replace unet and optim

    def gather(self, consts: torch.Tensor, t: torch.Tensor):
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1)

    def eval(self):
        self.unet.eval()

    def q_xt_x0(self, x0, t):
        mean = self.gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - self.gather(self.alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps

    def learn(self, input_trajs):
        x0 = input_trajs.to(self.device)
        t = torch.randint(low=0, high=self.n_steps, size=(len(x0) // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[: len(x0)]
        xt, noise = self.q_xt_x0(x0, t)
        pred_noise = self.unet(xt.float(), t)
        loss_eps = F.mse_loss(noise.float(), pred_noise)
        x0_hat = self.predict_x0(xt, t, pred_noise)
        recon_mse = F.mse_loss(x0_hat, x0)
        lam = getattr(cfg, "lambda_x0", 0.0)
        loss = loss_eps + lam * recon_mse
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), recon_mse.item()

    @torch.no_grad()
    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred_eps: torch.Tensor | None = None) -> torch.Tensor:
        if pred_eps is None:
            pred_eps = self.unet(xt.float(), t)
        sqrt_ab = self.gather(self.alpha_bar, t).sqrt()
        sqrt_one_minus_ab = (1.0 - self.gather(self.alpha_bar, t)).sqrt()
        return (xt - sqrt_one_minus_ab * pred_eps) / (sqrt_ab + 1e-8)

    def ddpm_step(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps_t = self.unet(xt.float(), t)
        alpha_t = self.gather(self.alpha, t)
        beta_t = self.gather(self.beta, t)
        sqrt_recip_alpha_t = (1.0 / (alpha_t + 1e-8)).sqrt()
        sqrt_one_minus_ab = (1.0 - self.gather(self.alpha_bar, t)).sqrt()
        mu_theta = sqrt_recip_alpha_t * (xt - (beta_t / (sqrt_one_minus_ab + 1e-8)) * eps_t)
        noise = torch.randn_like(xt) * (beta_t.sqrt())
        mask = (t > 0).float().reshape(-1, 1, 1)
        return mu_theta + mask * noise

    @torch.no_grad()
    def denoise_from(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = xt.size(0)
        cur = xt
        max_t = int(t.max().item())
        for step in range(max_t, -1, -1):
            step_t = torch.full((B,), step, device=xt.device, dtype=t.dtype)
            update_mask = (t >= step).float().reshape(-1, 1, 1)
            x_prev = self.ddpm_step(cur, step_t)
            cur = update_mask * x_prev + (1.0 - update_mask) * cur
        return cur

    @torch.no_grad()
    def infer_from_noise(self, input):
        B, C, L = input.shape
        xT = torch.randn(B, C, L, device=self.device)
        tT = torch.full((B,), self.n_steps - 1, device=self.device, dtype=torch.long)
        return self.denoise_from(xT, tT)

    def save_checkpoint(self, ckpt_path: str, epoch: int | None = None, metrics: dict | None = None, **_) -> str:
        unet = self.unet.module if hasattr(self.unet, "module") else self.unet
        ckpt = {
            "model": unet.state_dict(),
            "epoch": int(epoch) if epoch is not None else None,
            "n_steps": int(self.n_steps),
            "diff_lr": float(self.lr),
            "metrics": metrics or {},
        }
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(ckpt_path) or ".") as f:
            tmp_path = f.name
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str, map_location=None, strict: bool = True, **_):
        if map_location is None:
            map_location = cfg.device
        ckpt = torch.load(ckpt_path, map_location=map_location)
        unet = self.unet.module if hasattr(self.unet, "module") else self.unet
        state_dict_to_load = ckpt.get("model", ckpt)
        if isinstance(state_dict_to_load, dict) and "model" in ckpt:
            res = unet.load_state_dict(state_dict_to_load, strict=strict)
            missing, unexpected = res.missing_keys, res.unexpected_keys
        else:
            res = unet.load_state_dict(ckpt, strict=strict)
            missing, unexpected = res.missing_keys, res.unexpected_keys
        return {"epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None, "missing_keys": missing, "unexpected_keys": unexpected}
