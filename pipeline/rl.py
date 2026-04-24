# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# --------- 1) 轻量序列编码器：支持 avgpool 或 transformer ----------
class SeqEncoder(nn.Module):
    def __init__(self, dim_in: int, d_model: int = 256, nhead: int = 4, nlayers: int = 2,
                 use_transformer: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_transformer = use_transformer
        self.input_proj = nn.Linear(dim_in, d_model)

        if use_transformer:
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model*4,
                                                   dropout=dropout, batch_first=True)
            self.tr = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
            self.norm = nn.LayerNorm(d_model)
        else:
            # 纯 MLP + 均值池化，超轻
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, dim_in]   或 [B, C, T] 也能兼容（会自动转成 [B, L, D]）
        return: [B, d_model]
        """
        if x.dim() != 3:
            raise ValueError(f"Expect [B,L,D] or [B,C,T], got {x.shape}")
        B, A, B_or_T = x.shape
        # 兼容 [B,C,T]：如果第二维明显更小则视为 C，转置成 [B,T,C]
        if A <= 16 and B_or_T > A:  # 粗略规则：像 [B,C,T]
            x = x.transpose(1, 2)    # -> [B,T,C]

        h = self.input_proj(x)       # [B, L, d_model]
        if self.use_transformer:
            h = self.tr(h)           # [B, L, d_model]
            h = h.mean(dim=1)        # 全局平均池化
            h = self.norm(h)
        else:
            h = self.mlp(h)          # [B, L, d_model]
            h = h.mean(dim=1)
            h = self.norm(h)
        return h  # [B, d_model]

# --------- 2) Tanh-Normal 的对数概率修正 ----------
def tanh_normal_log_prob(mean: torch.Tensor, log_std: torch.Tensor, action_tanh: torch.Tensor) -> torch.Tensor:
    """
    mean/log_std: [B, act_dim]  正态参数（未tanh）
    action_tanh : [B, act_dim]  已经经过 tanh 的动作（-1,1）
    return      : [B]           每个样本的 sum(log_prob)
    """
    std = log_std.exp().clamp(min=1e-6)
    # 反tanh回未压缩空间
    # atanh(y) = 0.5 * ln((1+y)/(1-y))
    eps = 1e-6
    y = action_tanh.clamp(-1+eps, 1-eps)
    pre_tanh = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh

    # Normal log_prob
    log_prob = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2 * log_std + math.log(2 * math.pi))  # [B, act_dim]

    # tanh 的雅可比修正：sum log(1 - tanh(x)^2)
    correction = torch.log(1 - y.pow(2) + 1e-6)
    log_prob = log_prob.sum(dim=-1) - correction.sum(dim=-1)
    return log_prob

def tanh_normal_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """
    近似熵：用 Normal 熵减去一个常数近似（常见做法也直接用 Normal 熵）。
    Normal 熵：0.5 * log(2πeσ^2)
    返回标量熵（批均值）
    """
    std = log_std.exp().clamp(min=1e-6)
    normal_entropy = 0.5 * torch.log(2 * torch.pi * torch.e * std * std)  # [B, act_dim]
    return normal_entropy.sum(dim=-1).mean()

# --------- 3) 策略网络：Actor-Critic（连续动作） ----------
class RLPolicy(nn.Module):
    def __init__(self, dim_in: int, act_dim: int, d_model: int = 256, hidden: int = 256,
                 use_transformer: bool = True,
                 log_std_min: float = -5.0, log_std_max: float = 2.0,
                 action_low: Optional[torch.Tensor] = None,
                 action_high: Optional[torch.Tensor] = None):
        super().__init__()
        self.encoder = SeqEncoder(dim_in, d_model=d_model, use_transformer=use_transformer)
        self.actor = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.logstd_head = nn.Linear(hidden, act_dim)

        self.critic = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 动作缩放到物理范围（可选）
        if action_low is not None and action_high is not None:
            assert action_low.shape == action_high.shape == (act_dim,)
            self.register_buffer("a_low", action_low)
            self.register_buffer("a_high", action_high)
            self.use_scale = True
        else:
            self.use_scale = False
            self.register_buffer("a_low", torch.tensor(-1.0).repeat(act_dim))
            self.register_buffer("a_high", torch.tensor( 1.0).repeat(act_dim))

        # 记录最近一次分布以便取熵（可选）
        self._last_mean = None
        self._last_log_std = None
        self._last_action_tanh = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, L, dim_in] 或 [B, C, T]
        return:
          action_tgt: [B, act_dim]  （已缩放到物理范围）
          logp      : [B]           对应动作的 log_prob
          value     : [B]           状态价值
        """
        feat = self.encoder(x)  # [B, d_model]
        value = self.critic(feat).squeeze(-1)  # [B]

        h = self.actor(feat)
        mean = self.mu_head(h)
        log_std = self.logstd_head(h).clamp(self.log_std_min, self.log_std_max)

        # 采样（reparameterization trick）
        std = log_std.exp().clamp(min=1e-6)
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        action_tanh = torch.tanh(pre_tanh)         # [-1,1]
        logp = tanh_normal_log_prob(mean, log_std, action_tanh)  # [B]

        # 物理范围缩放
        if self.use_scale:
            action = self._scale_to_range(action_tanh)  # [B, act_dim]
        else:
            action = action_tanh

        # 缓存
        self._last_mean = mean.detach()
        self._last_log_std = log_std.detach()
        self._last_action_tanh = action_tanh.detach()

        return action, logp, value

    @torch.no_grad()
    def act_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        h = self.actor(feat)
        mean = self.mu_head(h)
        action_tanh = torch.tanh(mean)
        if self.use_scale:
            return self._scale_to_range(action_tanh)
        return action_tanh

    def dist_entropy(self) -> torch.Tensor:
        # 近似用 Normal 熵
        if self._last_log_std is None:
            return torch.tensor(0.0, device=self.mu_head.weight.device)
        return tanh_normal_entropy(self._last_log_std)

    def _scale_to_range(self, a_tanh: torch.Tensor) -> torch.Tensor:
        # [-1,1] -> [low, high]
        scale = (self.a_high - self.a_low) / 2.0
        bias = (self.a_high + self.a_low) / 2.0
        return a_tanh * scale + bias

# --------- 4) 你要的入口：get_rl_extra_info ----------
def get_rl_action(rl: RLPolicy, info_pre: torch.Tensor, eval_mode: bool = False):
    """
    info_pre: [B, L, dim_in] 或 [B, C, T]
    返回: (action, logp, value)
      - 训练时 eval_mode=False：采样动作 + 对应logp、value
      - 推理时 eval_mode=True ：确定性动作（mean经过tanh），logp 置 0（不用于优化）
    """
    info_pre = torch.transpose(info_pre, 1, 2)
    if eval_mode:
        with torch.no_grad():
            action = rl.act_deterministic(info_pre)
            value = rl.critic(rl.encoder(info_pre)).squeeze(-1)
        logp = torch.zeros(action.size(0), device=action.device)
        return action, logp, value
    else:
        return rl(info_pre)  # (action, logp, value)
