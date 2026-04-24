import torch
import torch.nn.functional as F

@torch.no_grad()
def debug_diffusion_once(model, info, x0, extra=None, seed=1234):
    was_training = model.unet.training
    model.unet.eval()

    B = x0.size(0)
    device = x0.device
    t = torch.randint(low=0, high=model.n_steps, size=(B,), device=device)

    # 前向加噪
    xt, true_noise = model.q_xt_x0(x0, t)

    # 预测噪声 & 单步反推
    eps_hat = model.unet(xt.float(), info, t, extra=extra)
    x0_hat  = model.predict_x0(xt, t, eps_hat)

    # 指标①②
    mse_eps = F.mse_loss(eps_hat.float(), true_noise.float()).item()
    mse_x0  = F.mse_loss(x0_hat, x0).item()

    # 构造 μθ（按你 ddpm_step 的公式）
    alpha_t = model.gather(model.alpha, t)
    beta_t  = model.gather(model.beta, t)
    abar_t  = model.gather(model.alpha_bar, t)
    sqrt_recip_alpha_t  = (1.0 / (alpha_t + 1e-8)).sqrt()
    sqrt_one_minus_ab_t = (1.0 - abar_t).clamp_min(1e-12).sqrt()
    mu_theta = sqrt_recip_alpha_t * (xt - (beta_t / (sqrt_one_minus_ab_t + 1e-8)) * eps_hat)

    # 跑你自己的 ddpm_step（会加噪音）
    x_prev = model.ddpm_step(xt, info, t, extra=extra)

    # 标准化残差（看噪声项是否“像N(0,β)”）
    sigma = beta_t.clamp_min(1e-12).sqrt()
    z = (x_prev - mu_theta) / (sigma + 1e-12)
    z_mean = z.mean().item()
    z_std  = z.std(unbiased=False).item()
    z_max  = z.abs().max().item()

    # 你现在用的是 √β vs 真实后验 √β̃ 的偏差
    t_minus1 = torch.clamp(t - 1, min=0)
    abar_tm1 = model.gather(model.alpha_bar, t_minus1)
    beta_tilde = beta_t * ((1.0 - abar_tm1).clamp_min(1e-12) / (1.0 - abar_t).clamp_min(1e-12))
    ratio_sigma = (beta_t.clamp_min(1e-12).sqrt() / beta_tilde.clamp_min(1e-12).sqrt()).mean().item()

    if was_training:
        model.unet.train()

    print({
        "eps_mse": round(mse_eps, 6),   # 预测噪声对不对
        "x0_mse":  round(mse_x0,  6),   # 单步反推是否准
        "z_mean":  round(z_mean,  6),   # 应≈0
        "z_std":   round(z_std,   6),   # 应≈1
        "z_max":   round(z_max,   6),
        "sigma/tilde_sigma": round(ratio_sigma, 6)  # 明显≠1 说明你用了错的方差
    })

@torch.no_grad()
def debug_diffusion_multi(model, info, x0, extra=None, runs=5):
    for s in range(1234, 1234 + runs):
        debug_diffusion_once(model, info, x0, extra=extra, seed=s)