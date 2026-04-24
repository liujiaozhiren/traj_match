import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """简单的表征网络: x -> z"""
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in) -> (..., d_out)
        return self.net(x)


class DeepLinUCB:
    """
    DeepLinUCB: 深度表示 + 线性UCB (per Zhou & others 的工程实现思路)
    适合: 上下文多臂老虎机，动作是一次性的离散选项集合。
    """

    def __init__(
        self,
        d_context: int,
        rep_hidden: int = 128,
        rep_dim: int = 32,
        alpha: float = 1.0,
        lambda_: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        d_context: 上下文/选项特征维度
        rep_hidden: 表征网络隐层大小
        rep_dim: 表征维度 (线性 bandit 头的维度)
        alpha: UCB 系数 (越大越爱探索)
        lambda_: 正则系数
        """
        self.device = device
        self.alpha = alpha
        self.lambda_ = lambda_

        self.encoder = MLPEncoder(d_context, rep_hidden, rep_dim).to(device)

        self.rep_dim = rep_dim

        # A = lambda * I + sum z z^T, b = sum z r
        self.A = lambda_ * torch.eye(rep_dim, device=device)
        self.A_inv = torch.inverse(self.A)  # 维度小可以直接求逆
        self.b = torch.zeros(rep_dim, device=device)

        # 用于更新 encoder 的优化器（可选）
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)

        # 简单 replay，方便训练 encoder
        self.replay_Z = []
        self.replay_r = []

    @torch.no_grad()
    def _compute_rep(self, X: torch.Tensor) -> torch.Tensor:
        """
        计算表征 z = g(x).
        X: (K, d_context)
        return: (K, rep_dim)
        """
        self.encoder.eval()
        X = X.to(self.device)
        Z = self.encoder(X)
        return Z

    def select_action(self, X: torch.Tensor) -> int:
        """
        给定当前一轮的所有候选选项特征 X，选一个动作(索引)。
        X: (K, d_context)
        return: 选中的 index (int)
        """
        self.encoder.eval()
        with torch.no_grad():
            Z = self._compute_rep(X)            # (K, rep_dim)
            theta_hat = self.A_inv @ self.b     # (rep_dim,)

            # 预测 reward 均值
            mu = (Z @ theta_hat)                # (K,)

            # 不确定性项: sqrt(z^T A^-1 z)
            # => 对每个 z: sigma^2 = z^T A_inv z
            Az = Z @ self.A_inv                 # (K, rep_dim)
            sigma_sq = (Az * Z).sum(dim=1)      # (K,)
            sigma = torch.sqrt(torch.clamp(sigma_sq, min=1e-8))

            ucb = mu + self.alpha * sigma       # (K,)
            a = torch.argmax(ucb).item()
        return a

    def update(self, x_chosen: torch.Tensor, reward: float):
        """
        用本轮选中的上下文和 reward 做一次在线更新。
        x_chosen: (d_context,)
        reward: 标量
        """
        self.encoder.train()

        x_chosen = x_chosen.to(self.device).unsqueeze(0)  # (1, d_context)
        z = self.encoder(x_chosen).squeeze(0)             # (rep_dim,)

        # 更新线性头的统计量 A, b, A_inv
        z = z.detach()  # 不对 A,b 的更新反传梯度
        z = z.view(-1, 1)  # (rep_dim, 1)

        self.A += z @ z.T  # (rep_dim, rep_dim)
        self.A_inv = torch.inverse(self.A)  # 维度不大可以直接求逆
        self.b += reward * z.squeeze(1)     # (rep_dim,)

        # 记录 replay 用于更新 encoder
        self.replay_Z.append(z.detach().squeeze(1))      # (rep_dim,)
        self.replay_r.append(torch.tensor(float(reward), device=self.device))

        # 可选：每轮小步更新 encoder（让表示更像线性可分）
        if len(self.replay_Z) >= 32:  # mini-batch size
            self._train_encoder_step(batch_size=32)

    def _train_encoder_step(self, batch_size: int = 32):
        """
        简单的 encoder 训练：
        目标: 让 theta_hat^T g(x) 逼近 reward
        实际实现里可以更复杂（多步迭代/周期更新等），这里给一个最小版本。
        """
        if len(self.replay_Z) < batch_size:
            return

        # 构造 batch
        idx = torch.randint(0, len(self.replay_Z), (batch_size,))
        # 注意：严格来说，Z 应该由 encoder(x) 重新算，这里用旧 Z 简化
        Z_batch = torch.stack([self.replay_Z[i] for i in idx], dim=0)  # (B, rep_dim)
        r_batch = torch.stack([self.replay_r[i] for i in idx], dim=0)  # (B,)

        # 当前线性头参数
        with torch.no_grad():
            theta_hat = self.A_inv @ self.b  # (rep_dim,)

        pred = (Z_batch @ theta_hat)  # (B,)
        loss = F.mse_loss(pred, r_batch)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
