import torch, cfg
import torch.nn as nn
import torch.nn.functional as F
import os
import tempfile

from my_diffusion.unet.assistTrajUnet import AssistTrajUnet
from my_diffusion.base.model import MyBaseDiff
from pretrain.code.DiffTraj.utils.EMA import EMAHelper, SafeEMA


class FixLenDiff(MyBaseDiff):

    def __init__(self, device=cfg.device, lr= 1e-4):
        super(FixLenDiff, self, ).__init__(device=device)

        self.unet = AssistTrajUnet().to(device)
        self.optim = torch.optim.AdamW(self.unet.parameters(), lr=lr)  # Optimizer
        if cfg.diff_ema:
            self.ema_helper = SafeEMA(mu=0.9999)
            self.ema_helper.register(self.unet)
        else:
            self.ema_helper = None


    def learn(self, input_trajs, lam=None):
        info = input_trajs[..., :cfg.diff_pre_len]
        target = input_trajs[..., cfg.diff_pre_len:]
        info = info.to(self.device)
        x0 = target.to(self.device)

        t = torch.randint(low=0, high=self.n_steps, size=(len(x0) // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:len(x0)]
        # Get the noised images (xt) and the noise (our target)
        xt, noise = self.q_xt_x0(x0, t)
        # Run xt through the network to get its predictions
        pred_noise = self.unet(xt.float(), info, t)

        loss_eps = F.mse_loss(noise.float(), pred_noise)

        # 新增：用预测噪声反推 \hat{x}_0，并与真 x0 做对比（仅作监控，或加入联合损失）
        x0_hat = self.predict_x0(xt, t, pred_noise)
        recon_mse = F.mse_loss(x0_hat, x0)

        # 是否把重建项合到总损失里，可用权重开关
        if lam is None:
            lam = getattr(cfg, "lambda_x0", 0.2)
        loss = loss_eps + lam * recon_mse

        # Store the loss for later viewing
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        if cfg.diff_ema:
            self.ema_helper.update(self.unet)
        return loss.item(), recon_mse.item()

    @torch.no_grad()
    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred_eps: torch.Tensor = None) -> torch.Tensor:
        """
        由 x_t 和 预测噪声 ε_θ 反推 \hat{x}_0:
          \hat{x}_0 = (x_t - sqrt(1-ᾱ_t)*ε_θ) / sqrt(ᾱ_t)
        维度自适应：gather(...)->(-1,1,1) 以兼容 [B,C,L] / [B,*,*]
        """
        if pred_eps is None:
            # pred_eps = self.unet(xt.float(), info, t)
            raise ValueError("pred_eps must be provided")
        sqrt_ab = self.gather(self.alpha_bar, t).sqrt()
        sqrt_one_minus_ab = (1. - self.gather(self.alpha_bar, t)).sqrt()
        x0_hat = (xt - sqrt_one_minus_ab * pred_eps) / (sqrt_ab + 1e-8)
        return x0_hat

    # ---- 新增：DDPM 单步反推 p_θ(x_{t-1}|x_t)（predict noise 参数化）----
    def ddpm_step(self, xt: torch.Tensor, info: torch.Tensor, t: torch.Tensor, extra=None) -> torch.Tensor:
        """
        μ_θ = 1/sqrt(α_t) * (x_t - β_t / sqrt(1-ᾱ_t) * ε_θ)
        x_{t-1} = μ_θ + σ_t * z,  σ_t = sqrt(β_t)（最基础设定）
        """
        eps_t = self.unet(xt.float(), info,  t, extra=extra)
        # print(f'alpha device:{self.alpha.device}')
        # print(f't device:{self.alpha.device}')
        alpha_t = self.gather(self.alpha, t)
        beta_t = self.gather(self.beta, t)
        sqrt_recip_alpha_t = (1.0 / (alpha_t + 1e-8)).sqrt()
        sqrt_one_minus_ab = (1. - self.gather(self.alpha_bar, t)).sqrt()

        mu_theta = sqrt_recip_alpha_t * (xt - (beta_t / (sqrt_one_minus_ab + 1e-8)) * eps_t)

        # t>0 才加噪；t==0 直接返回均值
        noise = torch.randn_like(xt) * (beta_t.sqrt())
        mask = (t > 0).float().reshape(-1, 1, 1)
        x_prev = mu_theta + mask * noise
        return x_prev

    # ---- 新增：从任意起点往回完整去噪（可用于推理/采样）----
    @torch.no_grad()
    def denoise_from(self, xt: torch.Tensor, info: torch.Tensor, t: torch.Tensor, extra=None) -> torch.Tensor:
        """
        给定一批 x_t 及各自 t，逐步跑到 t=0 返回近似 x_0。
        若想纯采样：传入 x_T ~ N(0,I) 且 t 全部为 T-1。
        """
        # 以每个样本自己的 t 开始往回
        # 为了简洁，这里按最大 t 往回，低于当前步的样本不再变化
        B = xt.size(0)
        cur = xt
        max_t = int(t.max().item())
        for step in range(max_t, -1, -1):
            step_t = torch.full((B,), step, device=xt.device, dtype=t.dtype)
            # 只更新那些原本 t>=step 的样本
            update_mask = (t >= step).float().reshape(-1, 1, 1)
            x_prev = self.ddpm_step(cur, info, step_t, extra=extra)
            cur = update_mask * x_prev + (1. - update_mask) * cur
        return cur  # 近似 x_0

    @torch.no_grad()
    def infer_from_noise(self, info, extra=None, num=None):
        B, C, _ = info.shape
        device = info.device
        L = cfg.diff_infer_len
        last_step = self.n_steps - 1
        if num is None:
            xT = torch.randn(B, C, L, device=device)
            tT = torch.full((B,), last_step, device=device, dtype=torch.long)
            x0_sample = self.denoise_from(xT, info, tT, extra=extra)
        else:
            num = int(num)
            assert num >= 1, f"num must be >=1, got {num}"
            # 批维重复 info
            info_rep = info.repeat_interleave(num, dim=0)
            xT = torch.randn(B * num, C, L, device=device)
            tT = torch.full((B * num,), last_step, device=device, dtype=torch.long)
            assert extra is None , "当 num>1 时，extra应当为空"
            # 去噪并 reshape 成 (B, num, C, L)
            x0 = self.denoise_from(xT, info_rep, tT, extra=extra)  # (B*num, C, L)
            x0_sample = x0.reshape(B, num, C, L)
        return x0_sample

    def save_checkpoint(self, ckpt_path: str, epoch: int = None,
                        metrics: dict | None = None,
                        save_optimizer: bool = True,
                        save_ema: bool = True) -> str:
        """
        保存当前unet权重；可选保存优化器与EMA。
        返回最终保存的文件路径。
        """
        # 处理 DDP/DataParallel 包裹
        unet = self.unet.module if hasattr(self.unet, "module") else self.unet

        ckpt = {
            "model": unet.state_dict(),
            "epoch": int(epoch) if epoch is not None else None,
            "n_steps": int(self.n_steps),
            "diff_lr": float(self.lr),
            "metrics": metrics or {},
            "pytorch_version": torch.__version__,
        }
        if save_optimizer and hasattr(self, "optim") and self.optim is not None:
            ckpt["optimizer"] = self.optim.state_dict()

        # 保存 EMA（若存在）
        ema_obj = getattr(self, "ema_helper", None)
        if save_ema and ema_obj is not None:
            # 兼容两种写法：有 state_dict() 或有 shadow 参数字典
            if hasattr(ema_obj, "state_dict"):
                ckpt["ema"] = {"type": "state_dict", "payload": ema_obj.state_dict()}
            elif hasattr(ema_obj, "shadow"):
                ckpt["ema"] = {"type": "shadow", "payload": ema_obj.shadow}
            else:
                ckpt["ema"] = None

        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        # 原子落盘：先写临时文件，再rename 避免中途损坏
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(ckpt_path) or ".") as f:
            tmp_path = f.name
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, ckpt_path)
        return ckpt_path

    # === 新增：读取 ===
    def load_checkpoint(self, ckpt_path: str,
                        map_location: str | torch.device | None = None,
                        strict: bool = True,
                        load_ema: bool = False) -> dict:
        """
        读取checkpoint；默认加载 model 权重。
        load_ema=True 时，如有EMA则优先把EMA权重拷到模型。
        返回：一些元信息（如 epoch/metrics）。
        """
        if map_location is None:
            map_location = getattr(torch.cuda, "current_device", lambda: "cpu")()
            # 若你用cfg.device，也可以：map_location = cfg.device

        ckpt = torch.load(ckpt_path, map_location=map_location)

        # 处理 DDP/DataParallel 包裹
        unet = self.unet.module if hasattr(self.unet, "module") else self.unet

        # 先决定加载哪个权重
        state_dict_to_load = ckpt.get("model", {})
        if load_ema and ckpt.get("ema"):
            ema_info = ckpt["ema"]
            if ema_info["type"] == "state_dict" and hasattr(getattr(self, "ema_helper", None), "load_state_dict"):
                # 1) 先把 EMA 自身恢复，再把 EMA 应用到模型（如果你的 EMAHelper 有 apply/to 方法）
                self.ema_helper.load_state_dict(ema_info["payload"])
                # 常见EMA工具会有 copy_to/apply_to；这里做个兼容：
                if hasattr(self.ema_helper, "copy_to"):
                    self.ema_helper.copy_to(unet)
                elif hasattr(self.ema_helper, "apply_to"):
                    self.ema_helper.apply_to(unet)
                else:
                    # 退化：直接取 shadow 覆盖
                    if hasattr(self.ema_helper, "shadow"):
                        unet.load_state_dict(self.ema_helper.shadow, strict=strict)
                    # 提前返回（已经用EMA拷了）
                    pass
                # 如果上面已经通过copy_to/apply_to等覆盖，就无需再 load_state_dict
                state_dict_to_load = None
            elif ema_info["type"] == "shadow":
                # 直接把 shadow 当作模型权重
                state_dict_to_load = ema_info["payload"]

        missing, unexpected = [], []
        if state_dict_to_load is not None:
            res = unet.load_state_dict(state_dict_to_load, strict=strict)
            # res 是一个 NamedTuple: (missing_keys, unexpected_keys)
            missing, unexpected = res.missing_keys, res.unexpected_keys

        # 恢复优化器（可选）
        if "optimizer" in ckpt and hasattr(self, "optim") and self.optim is not None:
            try:
                self.optim.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                # 可能因为参数名不一致（比如换了优化器/参数组）
                print(f"[load_checkpoint] optimizer state not loaded: {e}")

        info = {
            "epoch": ckpt.get("epoch"),
            "metrics": ckpt.get("metrics", {}),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "pytorch_version": ckpt.get("pytorch_version"),
        }
        return info