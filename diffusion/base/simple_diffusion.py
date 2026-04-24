# Refactored diffusion model as a single class
import math, torch, torch.nn as nn, torch.nn.functional as F

class DenoisingDiffusionTS(nn.Module):
    """
    Minimal DDPM for (L,d) trajectories, wrapped in one class.
    ----------------------------------------------------------
    Methods
    -------
    * q_sample(x0, t)         : forward corruption
    * p_sample_step(xt, t)    : DDPM reverse one step
    * sample(n)               : generate n trajectories
    * fit(loader, epochs)     : one-line training loop

    Example
    -------
    >>> model = DenoisingDiffusionTS(L=128, d=4, T=1000).to('cuda')
    >>> model.fit(train_loader, epochs=10)
    >>> x_gen = model.sample(8)
    """

    def __init__(self, L: int, d: int, T: int = 1000,
                 hidden: int = 64, device: str = 'cuda'):
        super().__init__()
        self.L, self.d, self.T = L, d, T
        self.device = device

        # ---- noise schedule & buffers ----
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)

        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('abar',   abar.to(device))
        self.register_buffer('sqrt_abar', abar.sqrt().to(device))
        self.register_buffer('sqrt_one_m_abar', (1-abar).sqrt().to(device))
        self.register_buffer('poster_var', (betas * (1-abar.roll(1,0)) / (1-abar)).to(device))
        self.poster_var[0] = 0.0

        # ---- tiny MLP backbone ----
        self.net = nn.Sequential(
            nn.Linear(d+8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d)
        )

    # ----- sinusoidal embed -----
    def _time_embed(self, t: torch.Tensor, half_dim: int = 4) -> torch.Tensor:
        freqs = torch.exp(torch.arange(half_dim, device=self.device)
                          * (-math.log(10000.0)/(half_dim-1)))
        angles = t.float()[:,None] * freqs[None,:]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)   # (B,8)

    # ----- q(x_t | x0) -----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        B = x0.size(0)
        return ( self.sqrt_abar[t].view(B,1,1) * x0
               + self.sqrt_one_m_abar[t].view(B,1,1) * noise )

    # ----- predict eps -----
    def _eps_theta(self, xt, t):
        emb = self._time_embed(t).unsqueeze(1).expand(-1, self.L, -1)
        inp = torch.cat([xt, emb], dim=-1)
        return self.net(inp)

    # ----- p(x_{t-1} | x_t) one step -----
    @torch.no_grad()
    def p_sample_step(self, x, t_idx):
        t = torch.full((x.size(0),), t_idx, device=self.device, dtype=torch.long)
        eps_hat = self._eps_theta(x, t)
        a_t   = self.alphas[t_idx]
        coef1 = 1.0 / math.sqrt(a_t)
        coef2 = self.betas[t_idx] / self.sqrt_one_m_abar[t_idx]
        mu = coef1 * (x - coef2 * eps_hat)
        if t_idx == 0:
            return mu
        noise = torch.randn_like(x)
        var = self.poster_var[t_idx].sqrt()
        return mu + var * noise

    # ----- generation -----
    @torch.no_grad()
    def sample(self, n=4):
        x = torch.randn(n, self.L, self.d, device=self.device)
        for t_idx in reversed(range(self.T)):
            x = self.p_sample_step(x, t_idx)
        return x.cpu()

    # ----- training core -----
    def _loss_on_batch(self, x0):
        x0 = x0.to(self.device)
        B  = x0.size(0)
        t  = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_hat = self._eps_theta(xt, t)
        return F.mse_loss(eps_hat, noise)

    # ----- simple fit loop -----
    def fit(self, loader, epochs=10, lr=2e-5):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for ep in range(1, epochs+1):
            self.train(); total=0; n=0
            for x in loader if isinstance(loader.dataset[0], tuple) else loader:
                x0 = x[0]
                loss = self._loss_on_batch(x0[0])
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); n+=1
            print(f'Epoch {ep:03d}  loss {total/n:.4f}')

# ----------------- quick demo -----------------
if __name__ == "__main__":
    # synth sine dataset
    import torch.utils.data as data
    import matplotlib
    matplotlib.use("TkAgg")
    import math, torch, matplotlib.pyplot as plt
    L, d, N = 128, 1, 512
    t = torch.linspace(0, 2 * math.pi, L)
    X = torch.stack([torch.sin(t + phi) for phi in torch.rand(N) * 2 * math.pi])
    X = X.unsqueeze(-1).repeat(1, 1, d)
    #X = X.unsqueeze(-1).repeat(1,1,d)       # (512,L,d)
    ds = data.TensorDataset(X)
    dl = data.DataLoader(ds, batch_size=32, shuffle=True)

    model = DenoisingDiffusionTS(L=L, d=d, T=100, device='cuda' if torch.cuda.is_available() else 'cpu')
    model.fit(dl, epochs=200, lr=2e-4)

    traj = model.sample(20)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    for k in range(d):
        ax[0].plot(X[0, :, k]);
        ax[1].plot(traj[0, :, k])
    ax[0].set_title("Input trajectory");
    ax[1].set_title("Generated")
    for a in ax:
        a.set_xticks([]);
        a.set_yticks([]);
        a.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

    # —— y 轴刻度：根据两图数据整体 min ~ max ——
    y_min = min(X.min(), traj.min()).item()
    y_max = max(X.max(), traj.max()).item()
    import numpy as np

    yticks = np.linspace(y_min, y_max, 5)  # 5 个等距刻度
    for a in ax:
        a.set_yticks(yticks)
        a.set_yticklabels([f"{y:.1f}" for y in yticks])

    plt.tight_layout();
    plt.show()
    print("Generated:", traj.shape)
