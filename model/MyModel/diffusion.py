"""
Triangular‑Noise DDPM (annotated)
================================
核心思路
--------
* **三角噪声块**  `Z_list[j]` 只有最后 *j* 行为 𝒩(0,1)，其余行为 0。
* **权重向量**  `w(t)` 让噪声沿行向上填充：
  前 *k* 行权重 1，第 *k* 行权重 *frac*∈(0,1)，更高行为 0。
* **正向 (q_sample)**  经典 DDPM：`x_t ← √ᾱ x0 + √(1-ᾱ) ε(t)`。
* **反向 (p_sample_step)**  预测 ε̂ → 计算 µ_t → 采样 `x_{t-1}`。

代码结构
--------
```
triangular_diffusion.py
├── hyper‑params & dataset
├── triangular noise utils  (weight  / triangular_noise)
├── ddpm util functions      (q_sample / p_sample_step)
├── simple 1‑D UNet backbone
├── train()                 # 1 epoch demo
└── sample()                # 反向链
```
运行
----
```bash
python triangular_diffusion.py
```
会打印 loss 并在训练后生成 4 条 (L,d) 轨迹矩阵。
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -------------------------------------------------------------
# 0. Hyper‑parameters & dummy data
# -------------------------------------------------------------
L, d   = 32, 2          # 轨迹长度 / 特征维度 (自行替换)
T      = 1000           # 扩散步数
BATCH  = 64             # mini‑batch
EPOCHS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

class DummyTraj(Dataset):
    """产生 (N,L,d) 的正弦波轨迹，可替换为真实数据集"""
    def __init__(self, N: int = 4096):
        super().__init__()
        t = torch.linspace(0, 2 * math.pi, L)
        waves = torch.stack([
            torch.stack([torch.sin(t * f), torch.cos(t * f)], 1)
            for f in (torch.rand(N) * 2.0 + 0.5)  # 随机频率
        ])
        self.data = waves.float()  # (N,L,d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data_loader = DataLoader(DummyTraj(), BATCH, shuffle=True, drop_last=True)

# -------------------------------------------------------------
# 1. 预生成三角噪声块  Z_list[j] ∈ ℝ^{L×d}
# -------------------------------------------------------------
Z_list = []
for j in range(1, L + 1):
    z = torch.zeros(L, d)
    z[L - j :, :] = torch.randn(j, d)  # 仅最后 j 行是高斯噪声
    Z_list.append(z.to(device))

# -------------------------------------------------------------
# 2. 标量 β‑schedule 及辅助张量
# -------------------------------------------------------------
beta = torch.linspace(1e-4, 0.02, T, device=device)  # 余弦 / 线性均可
alpha = 1 - beta
abar = torch.cumprod(alpha, 0)  # (T,)

# -------------------------------------------------------------
# 3. Triangular‑noise helpers
# -------------------------------------------------------------

def weight(t: int) -> torch.Tensor:
    """生成权重向量 w(t) ∈ ℝ^L : [1…1, frac, 0…0]"""
    s = (t + 1) * L / T           # 连续进度
    k = int(s)                    # 已填满行数
    frac = s - k                  # 当前行的填充比例
    w = torch.zeros(L, device=device)
    w[:k] = 1.0
    if k < L:
        w[k] = frac
    return w                      # (L,)

@torch.no_grad()
def triangular_noise(t: int, batch: int) -> torch.Tensor:
    """按权重叠加三角噪声块, 返回 ε_t ∈ ℝ^{B×L×d}"""
    w = weight(t).view(L, 1)          # (L,1) broadcast 到 (L,d)
    eps_single = sum(w[j] * Z_list[j] for j in range(L))  # (L,d)
    return eps_single.unsqueeze(0).expand(batch, L, d)    # 复制到 batch

# -------------------------------------------------------------
# 4. Forward   q_sample(x0,t)  &   Reverse   p_sample_step
# -------------------------------------------------------------

def q_sample(x0: torch.Tensor, t_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """正向加噪: 从 x0 得到 x_t 和对应 ε_t (batch 同步)"""
    B = x0.size(0)
    eps = torch.stack([triangular_noise(ti.item(), 1)[0] for ti in t_idx])  # (B,L,d)
    sqrt_ab = abar[t_idx].sqrt().view(B, 1, 1)
    sqrt_mab = (1 - abar[t_idx]).sqrt().view(B, 1, 1)
    xt = sqrt_ab * x0 + sqrt_mab * eps
    return xt, eps

def p_sample_step(model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
    """反向一步 x_t → x_{t-1} (DDPM 标准公式)"""
    B = x_t.size(0)
    t_batch = torch.full((B,), t, device=device, dtype=torch.long)
    eps_hat = model(x_t.permute(0, 2, 1), timestep_embed(t_batch)).permute(0, 2, 1)

    a_t, ab_t = alpha[t], abar[t]
    coef1 = 1 / math.sqrt(a_t)
    coef2 = (1 - a_t) / math.sqrt(1 - ab_t)
    mu = coef1 * (x_t - coef2 * eps_hat)
    if t > 0:
        sigma = math.sqrt(beta[t])
        x_prev = mu + sigma * torch.randn_like(x_t)
    else:
        x_prev = mu
    return x_prev

# -------------------------------------------------------------
# 5. Sinusoidal timestep embedding
# -------------------------------------------------------------

def timestep_embed(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / (half - 1)))
    ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B,dim)

# -------------------------------------------------------------
# 6. 极简 1‑D UNet backbone
# -------------------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.norm = nn.GroupNorm(4, out_c)
        self.pool = nn.AvgPool1d(2)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.pool(self.act(self.norm(self.conv(x))))

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.norm = nn.GroupNorm(4, out_c)
        self.up   = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.act  = nn.SiLU()
    def forward(self, x):
        return self.act(self.norm(self.conv(self.up(x))))

class UNet1D(nn.Module):
    """(B,d,L) → (B,d,L) 预测 ε_hat"""
    def __init__(self, base: int = 64, t_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, base), nn.SiLU(), nn.Linear(base, base)
        )
        self.d1 = DownBlock(d, base)
        self.d2 = DownBlock(base, base * 2)
        self.mid = nn.Sequential(
            nn.Conv1d(base * 2, base * 2, 3, padding=1), nn.SiLU(),
            nn.Conv1d(base * 2, base * 2, 3, padding=1)
        )
        self.u1 = UpBlock(base * 2, base)
        self.u2 = UpBlock(base, base)
        self.out = nn.Conv1d(base, d, 1)
    def forward(self, x, t_embed):
        t_feat = self.time_mlp(t_embed)[:, :, None]  # (B,base,1)
        d1 = self.d1(x)
        d2 = self.d2(d1) + t_feat
        m  = self.mid(d2)
        u1 = self.u1(m) + d1
        u2 = self.u2(u1)
        return self.out(u2)

model = UNet1D().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=2e-4)

# -------------------------------------------------------------
# 7. Training loop (三行→函数化)
# -------------------------------------------------------------

def train_one_epoch(epoch: int):
    model.train()
    pbar = tqdm(data_loader, desc=f"epoch {epoch}")
    for x0 in pbar:
        x0 = x0.to(device)                   # (B,L,d)
        B = x0.size(0)
        t_idx = torch.randint(0, T, (B,), device=device)
        xt, eps = q_sample(x0, t_idx)        # 正向加噪
        eps_hat = model(xt.permute(0, 2, 1), timestep_embed(t_idx)).permute(0, 2, 1)
        loss = ((eps_hat - eps) ** 2).mean()
        optim.zero_grad(); loss.backward(); optim.step()
        pbar.set_postfix(loss=loss.item())

@torch.no_grad()
def sample(n: int = 4) -> torch.Tensor:
    """从纯噪声逐步反向生成,"""
    x_t = torch.randn(n, L, d, device=device)
    for t in reversed(range(T)):
        x_t = p_sample_step(model, x_t, t)
    return x_t.cpu()

if __name__ == "__main__":
    for e in range(1, EPOCHS + 1):
        train_one_epoch(e)

    traj = sample()
    print("generated", traj.shape)
    Path("out").mkdir(exist_ok=True)
    torch.save(traj, "out/generated.pt")
    print("Saved to out/generated.pt")
