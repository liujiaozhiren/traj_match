import torch
import cfg
import torch.nn as nn

from my_diffusion.base.model import MyBaseDiff
from my_diffusion.unet.assistTrajPointUnet import AssistTrajUnetPivot
from pretrain.code.DiffTraj.utils.EMA import EMAHelper


class StepwiseForwardDiffTail(MyBaseDiff):
    """
    逐点扩散（仅最后 diff_infer_len 个点），只实现前向 q(x_t|x_0)：
      - L 固定为 50
      - 只在 j ∈ [L-diff_infer_len, L-1] 的位置上逐点加噪
      - 总步数 T=diffusion_timestamp，均匀分配成每点 k=T//diff_infer_len 个“微步”
      - t∈[0, k*diff_infer_len-1]，映射到 pivot 位置 p(t)=L-1-⌊t/k⌋，微步 s(t)=t mod k
    """
    def __init__(self, device=cfg.device):
        super(StepwiseForwardDiffTail, self).__init__()
        self.L = 50
        self.infer_len = int(cfg.diff_infer_len)              # 例如 8
        self.T_total = int(cfg.diffusion_timestamp)           # 总步数
        assert self.infer_len >= 1 and self.infer_len <= self.L
        # 每个被扩散的位置的微步数 k（向下取整）
        self.k = max(1, self.T_total // self.infer_len)
        # 若有余数，可按需分配给靠后的若干位置（此处先忽略余数）
        self.T = self.k * self.infer_len

        self.device = torch.device(cfg.device if device is None else device)
        # 位置内的微步噪声调度（线性例子；可换成 cosine 等）
        beta_start = float(cfg.beta_start)
        beta_end   = float(cfg.beta_end)
        beta_k = torch.linspace(beta_start, beta_end, self.k, device=self.device)
        self.beta_k = beta_k                      # (k,)
        self.alpha_k = 1.0 - beta_k               # (k,)
        self.alpha_bar_k = torch.cumprod(self.alpha_k, dim=0)  # (k,)
        self.alpha_bar_full = self.alpha_bar_k[-1]             # 标量

        # 便捷索引
        self.j_start = self.L - self.infer_len    # 开始加噪的位置索引（含）
        self.unet = AssistTrajUnetPivot().to(cfg.device)

        self.optim = torch.optim.AdamW(self.unet.parameters(), lr=self.lr)  # Optimizer
        if cfg.diff_ema:
            self.ema_helper = EMAHelper(mu=0.9999)
            self.ema_helper.register(self.unet)
        else:
            self.ema_helper = None

    def learn(self, input_trajs):
        """
        训练一步
        input_trajs: (B, 2, 50) —— 全量轨迹（度量单位不影响训练）
        假设：
          - q_xt_x0(x0_full, t) 已实现：只对最后 infer_len 列加噪
          - build_predict_inputs(xt, t) 已实现：返回 ctx(42), pivot_val(1), local_t, p_idx
          - self.predict_eps(ctx, pivot_val, local_t) -> (B, 2, 1)：只预测当前 pivot 的噪声
        """
        x0_full = input_trajs.to(self.device)  # (B,2,50)
        B, C, L = x0_full.shape
        assert L == self.L == 50, f"Expect (B,2,50), got {tuple(x0_full.shape)}"

        # 1) 采样全局步 t（仅有效步，丢弃余数）
        T_eff = self.T  # = self.k * self.infer_len
        t = torch.randint(0, T_eff, (B,), device=self.device, dtype=torch.long)

        # 2) 前向加噪，得到 xt、整条 eps、按位置累计系数 abar
        xt, eps, abar = self.q_xt_x0(x0_full, t)  # xt/eps: (B,2,50), abar: (B,1,50)

        # 3) 组装 pivot-only 的输入（ctx: 从 pivot 往前 42 列，不含 pivot）
        ctx, pivot_val, local_t, p_idx = self.build_predict_inputs(xt, t)
        #    ctx      : (B,2,42)
        #    pivot_val: (B,2,1)  —— x_t 在 pivot 列的值
        #    local_t  : (B,)     —— s = t % k
        #    p_idx    : (B,)     —— 每个样本的 pivot 列索引

        # 4) 预测 pivot 的噪声
        eps_pred = self.predict_eps(ctx, pivot_val, local_t)  # (B,2,1)

        # 5) 取出真值噪声（只在 pivot 列监督）
        gather_idx = p_idx.view(B, 1, 1).expand(B, C, 1)  # (B,2,1)
        eps_true_pivot = torch.gather(eps, dim=2, index=gather_idx)  # (B,2,1)

        # 6) 噪声回归损失（pivot-only）
        loss_eps = torch.mean((eps_pred - eps_true_pivot) ** 2)

        # 7) 可选：pivot 列的 x0 重建正则
        lam = float(getattr(cfg, "lambda_x0", 0.0))
        if lam > 0.0:
            abar_pivot = torch.gather(abar, dim=2, index=p_idx.view(B, 1, 1))  # (B,1,1)
            x0_hat_pivot = (pivot_val - (1.0 - abar_pivot).sqrt() * eps_pred) / (abar_pivot.sqrt() + 1e-8)
            x0_true_pivot = torch.gather(x0_full, dim=2, index=gather_idx)  # (B,2,1)
            recon_mse = torch.mean((x0_hat_pivot - x0_true_pivot) ** 2)
        else:
            recon_mse = torch.tensor(0.0, device=self.device)

        loss = loss_eps + lam * recon_mse

        # 8) 反传 & 优化 & EMA
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        if self.ema_helper is not None:
            self.ema_helper.update(self.unet)

        return float(loss.item()), float(recon_mse.item())
    # 映射全局步 t -> (pivot 位置 p, 微步 s)
    def _split_t(self, t_global: torch.Tensor):
        """
        t_global: (B,) with 0..T-1
        p(t)=L-1 - floor(t/k), 仅落在最后 infer_len 个位置
        s(t)=t % k
        """
        t_global = torch.clamp(t_global, 0, self.T - 1)
        s = (t_global % self.k).long()
        block = (t_global // self.k).long()                 # 0..infer_len-1
        p = (self.L - 1) - block                            # 从末尾往前
        return p, s                                         # (B,), (B,)

    def alpha_bar_grid(self, x0: torch.Tensor, t_global: torch.Tensor):
        """
        生成位置级累计系数 \bar{alpha}(t)，形状 (B,1,L)：
          - j < j_start：恒 1（不加噪）
          - j > p(t) 且 j >= j_start：已满 k 步，取 alpha_bar_full
          - j = p(t)：取 alpha_bar_k[s(t)]
          - j_start <= j < p(t)：尚未开始，取 1
        """
        B, C, L = x0.shape
        assert L == self.L, f"Expect L={self.L}, got L={L}"
        device = x0.device
        p, s = self._split_t(t_global)                      # (B,), (B,)

        pos = torch.arange(L, device=device).view(1, 1, L)  # (1,1,L)
        p_grid = p.view(B, 1, 1)                            # (B,1,1)

        abar = torch.ones(B, 1, L, device=device)

        # 只在尾窗内计算
        in_window = (pos >= self.j_start)
        full_mask = in_window & (pos > p_grid)
        cur_mask  = in_window & (pos == p_grid)

        abar = abar.masked_fill(full_mask, self.alpha_bar_full)
        abar_cur = self.alpha_bar_k[s].view(B, 1, 1)        # (B,1,1)
        abar = abar * (~cur_mask) + abar_cur * cur_mask
        return abar

    def q_xt_x0(self, x0: torch.Tensor, t_global: torch.Tensor):
        """
        前向边缘采样：
          xt = sqrt(abar)*x0 + sqrt(1-abar)*eps,  eps ~ N(0,I)
        仅尾窗的列会随 t 变化；头部 (0..j_start-1) 恒等于 x0。
        返回 (xt, eps, abar) 便于调试。
        """
        x0 = x0.to(self.device)
        t_global = t_global.to(self.device)
        abar = self.alpha_bar_grid(x0, t_global)            # (B,1,L)
        eps  = torch.randn_like(x0)                         # (B,D,L)
        xt   = abar.sqrt() * x0 + (1.0 - abar).sqrt() * eps
        return xt, eps, abar


    """
    反向（去噪）最小实现，按 pivot-only 更新：
      - 只更新当前 pivot 列，其它列本步保持不变
      - predict_eps 的输入/输出按你的约定（ctx 42 列 + pivot 值 + 局部微步）
    """

    # ----------------- 1) 当步系数网格：仅 pivot 非平凡 -----------------
    def alpha_beta_step_grid(self, xt: torch.Tensor, t_global: torch.Tensor):
        """
        返回 alpha_t, beta_t，形状 (B,1,L)；仅 pivot 列非平凡：
          j==p: alpha_t=alpha_k[s], beta_t=beta_k[s]
          其他: alpha_t=1,        beta_t=0
        """
        B, C, L = xt.shape
        device = xt.device
        p, s = self._split_t(t_global)            # (B,), (B,)
        pos = torch.arange(self.L, device=device).view(1, 1, self.L)
        p_grid = p.view(B, 1, 1)
        cur = (pos == p_grid)

        ak = self.alpha_k[s].view(B, 1, 1)        # (B,1,1)
        bk = self.beta_k[s].view(B, 1, 1)         # (B,1,1)

        alpha_t = torch.ones(B, 1, self.L, device=device)
        beta_t  = torch.zeros(B, 1, self.L, device=device)
        alpha_t = alpha_t * (~cur) + ak * cur
        beta_t  = beta_t  * (~cur) + bk * cur
        return alpha_t, beta_t

    # ----------------- 2) 上一时刻累计系数：用于后验方差 -----------------
    def alpha_bar_prev_grid(self, xt: torch.Tensor, t_global: torch.Tensor):
        """
        返回 \bar{alpha}_{t-1} 的网格，形状 (B,1,L)。
        做法：先算 \bar{alpha}_t，再在 pivot 列除以 alpha_t，其它列不变。
        """
        abar_t = self.alpha_bar_grid(xt, t_global)         # (B,1,L)
        alpha_t, _ = self.alpha_beta_step_grid(xt, t_global)
        # 数值稳定：仅在 pivot 处 alpha_t<1，需要除法；其余 alpha_t=1
        abar_prev = abar_t / (alpha_t + 1e-8)
        return abar_prev

    # ----------------- 3) 组装 predict_eps 的输入（按你规定的签名） -----------------
    def build_predict_inputs(self, xt: torch.Tensor, t_global: torch.Tensor):
        """
        构造 (ctx, pivot_val, local_t)：
          ctx      : (B, D, 42)  = 从 pivot 往前数 42 列，不含 pivot
          pivot_val: (B, D, 1)   = 当前 pivot 列
          local_t  : (B,)        = 局部微步 s = t % k
        * 假定 L=50 且 infer_len<=8，保证 pivot-42 >= 0；否则这里会抛断言。
        """
        B, C, L = xt.shape
        assert L == self.L == 50, f"Expect L=50, got {L}"
        p, s = self._split_t(t_global)                      # (B,), (B,)

        # 计算每个样本的 ctx/pivot 切片
        ctx_list, piv_list = [], []
        for b in range(B):
            pb = int(p[b].item())
            start = pb - 42
            end   = pb
            if start < 0:
                raise ValueError(f"pivot={pb} 导致 ctx start<0；请确保 infer_len<=8 或改动窗口构造逻辑")
            ctx_list.append(xt[b:b+1, :, start:end])       # (1,D,42)
            piv_list.append(xt[b:b+1, :, pb:pb+1])         # (1,D,1)

        ctx = torch.cat(ctx_list, dim=0)                   # (B,D,42)
        pivot_val = torch.cat(piv_list, dim=0)             # (B,D,1)
        local_t = (t_global % self.k).long()               # (B,)
        return ctx, pivot_val, local_t, p                  # 额外返回 p 便于后续 scatter

    def predict_eps(self, ctx: torch.Tensor, pivot_val: torch.Tensor, local_t: torch.Tensor):
        """
        你需要实现/注入的噪声预测器：
          输入:
            ctx       (B, D, 42)   —— 倒数第 p 个点往前 42 列，不含 pivot
            pivot_val (B, D, 1)    —— 当前 pivot 列的值
            local_t   (B,)         —— 该点的局部微步索引 s∈[0,k-1]
          输出:
            eps_pivot (B, D, 1)    —— 仅当前 pivot 的噪声预测
        """
        return self.unet(pivot_val, ctx, local_t)
        # raise NotImplementedError("请在此处实现你的 eps 预测网络/函数")

    # ----------------- 4) 后验均值/方差（仅 pivot 列非平凡） -----------------
    def posterior_mean_variance_pivot(self, xt: torch.Tensor, t_global: torch.Tensor, eps_pivot: torch.Tensor, p_idx: torch.Tensor):
        """
        只计算 pivot 列的 \mu_\theta 与 \tilde\beta_t：
          mu_pivot  = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-abar_t) * eps_pred)  @pivot
          var_pivot = beta_t * (1-abar_{t-1})/(1-abar_t)                          @pivot
        返回:
          mu_pivot  : (B, D, 1)
          var_pivot : (B, 1, 1)   —— 每列各通道同方差，按位置广播
        """
        B, C, L = xt.shape
        device = xt.device
        alpha_t, beta_t = self.alpha_beta_step_grid(xt, t_global)   # (B,1,L)
        abar_t  = self.alpha_bar_grid(xt, t_global)                 # (B,1,L)
        abar_tm1 = self.alpha_bar_prev_grid(xt, t_global)           # (B,1,L)

        # 取 pivot 列的标量/向量
        gather_idx = p_idx.view(B, 1, 1).expand(B, 1, 1)            # (B,1,1)
        a_t   = torch.gather(alpha_t, dim=2, index=gather_idx)      # (B,1,1)
        b_t   = torch.gather(beta_t,  dim=2, index=gather_idx)      # (B,1,1)
        ab_t  = torch.gather(abar_t,  dim=2, index=gather_idx)      # (B,1,1)
        ab_tm1= torch.gather(abar_tm1,dim=2, index=gather_idx)      # (B,1,1)

        x_pivot = xt.gather(dim=2, index=gather_idx.expand(B, C, 1))  # (B,D,1)

        sqrt_recip_a = (1.0 / (a_t + 1e-8)).sqrt()                  # (B,1,1)
        sqrt_1m_ab   = (1.0 - ab_t).sqrt()                          # (B,1,1)

        mu_pivot = sqrt_recip_a * (x_pivot - (b_t / (sqrt_1m_ab + 1e-8)) * eps_pivot)  # (B,D,1)
        var_pivot = b_t * (1.0 - ab_tm1) / (1.0 - ab_t + 1e-8)                           # (B,1,1)
        return mu_pivot, var_pivot

    # ----------------- 5) 单步去噪（仅更新 pivot 列；t==0 不加噪） -----------------
    @torch.no_grad()
    def ddpm_step(self, xt: torch.Tensor, t_global: torch.Tensor):
        """
        从 x_t 得到 x_{t-1}（仅 pivot 列更新，其它列不变）。
        """
        ctx, pivot_val, local_t, p_idx = self.build_predict_inputs(xt, t_global)   # ctx:(B,D,42), pivot:(B,D,1), local:(B,)
        eps_pivot = self.predict_eps(ctx, pivot_val, local_t)                      # (B,D,1)

        mu_pivot, var_pivot = self.posterior_mean_variance_pivot(xt, t_global, eps_pivot, p_idx)  # (B,D,1), (B,1,1)

        # t==0 不再加噪
        noise_mask = (t_global > 0).view(-1, 1, 1).float()
        z = torch.randn_like(mu_pivot)
        x_prev_pivot = mu_pivot + noise_mask * var_pivot.sqrt() * z               # (B,D,1)

        # 写回到整条序列（只覆盖 pivot 列）
        x_prev = xt.clone()
        for b in range(xt.size(0)):
            pb = int(p_idx[b].item())
            x_prev[b, :, pb:pb+1] = x_prev_pivot[b:b+1]
        return x_prev

    @torch.no_grad()
    def init_xt_from_info(self, info: torch.Tensor):
        """
        info: (B,2,42) —— 已知前缀（真值）
        返回:
          xt_init : (B,2,50) —— 前42为info，后8为N(0,1)噪声
          t_start : (B,)     —— 全部从 T-1 开始去噪
        """
        info = info.to(self.device)
        B, C, P = info.shape
        assert C == 2 and P == (self.L - self.infer_len) == 42, f"Expect (B,2,42), got {tuple(info.shape)}"

        xt = torch.zeros(B, C, self.L, device=self.device, dtype=info.dtype)
        xt[:, :, :P] = info
        xt[:, :, P:] = torch.randn(B, C, self.infer_len, device=self.device, dtype=info.dtype)

        t_start = torch.full((B,), self.T - 1, device=self.device, dtype=torch.long)  # T = k * infer_len
        return xt, t_start

    @torch.no_grad()
    def infer_from_noise(self, info: torch.Tensor):
        """
        输入 (B,2,42) 的已知前缀，输出完整还原后的 (B,2,50)。
        去噪过程中每一步都强制把前42列钳住为info，确保上下文不被改动。
        """
        xt, t_start = self.init_xt_from_info(info)  # (B,2,50), (B,)
        B = xt.size(0)
        cur = xt

        # 从 T-1 -> 0 逐步去噪（只更新 pivot 列；前42列每步都钳住）
        for step in range(self.T - 1, -1, -1):
            step_t = torch.full((B,), step, device=self.device, dtype=torch.long)
            x_prev = self.ddpm_step(cur, step_t)  # 只会更新当前 pivot 列
            x_prev[:, :, :42] = info  # 关键：钳住已知前缀为真值
            cur = x_prev

        return cur[:,:,42:]  # (B,2,8)

# ================== main：只调试正向（尾部窗口版） ==================
if __name__ == "__main__":

    torch.manual_seed(0)
    fwd = StepwiseForwardDiffTail(cfg)

    B, D, L = 1, 2, 50
    # 构造一个简单的 x0 方便观察
    x0 = torch.stack([
        torch.linspace(0, 49, L),           # 通道0
        torch.linspace(100, 149, L),        # 通道1
    ], dim=0).unsqueeze(0).float()          # (1,2,50)

    print(f"L={L}, infer_len={cfg.diff_infer_len}, T={cfg.diffusion_timestamp} -> k={fwd.k}, T'={fwd.T}")
    print(f"尾窗范围: j ∈ [{fwd.j_start}, {L-1}]（只这部分会被加噪）\n")

    # 挑几步看看：每个块的第 0 步、最后一步；以及跨块的切换
    steps_to_probe = []
    for blk in [0, 1, cfg.diff_infer_len-1]:
        base = blk * fwd.k
        steps_to_probe += [base+0, base+fwd.k-1]
    steps_to_probe = [t for t in steps_to_probe if 0 <= t < fwd.T]
    steps_to_probe = sorted(set(steps_to_probe))

    for t in steps_to_probe:
        t_tensor = torch.tensor([t], dtype=torch.long)
        p, s = fwd._split_t(t_tensor)
        p, s = int(p.item()), int(s.item())

        xt, eps, abar = fwd.q_xt_x0(x0, t_tensor)
        xt_formula = abar.sqrt()*x0 + (1.0 - abar).sqrt()*eps
        max_err = (xt - xt_formula).abs().max().item()

        # 哪些列本步有噪声因子(1-abar)>0？
        noisy_cols = [j for j, v in enumerate((1.0 - abar[0,0]).tolist()) if v > 0]
        print(f"[t={t:3d}] pivot p={p:2d}, s={s:2d} | noisy_cols={noisy_cols[:8]}... (total {len(noisy_cols)}) | max|xt-xt_formula|={max_err:.2e}")