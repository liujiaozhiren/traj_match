# === rl_selector.py ==========================================================
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg
from pipeline.downstream import downstream_fn


# -------------------- config --------------------
class RLConfig:
    def __init__(
        self,
        algo: str = "ppo",        # "bandit" | "a2c" | "ppo"
        k_select: int = 4,        # 每个样本选择的对数
        hidden: int = 256,
        ent_coef: float = 1e-3,
        vf_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        # PPO 专用
        clip_eps: float = 0.2,
        ppo_epochs: int = 3,
        ppo_minibatches: int = 1,  # 单步场景通常就 1
    ):
        self.algo = algo
        self.k_select = k_select
        self.hidden = hidden
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.ppo_minibatches = ppo_minibatches

# -------------------- modules --------------------
class PairPolicy(nn.Module):
    """
    目标：在 N 维上选择。
    思路：每个候选 cand_(pre/post) 先展平 -> 低维投影 -> 用 info_f 产生 FiLM(gamma,beta) 做条件化
         -> pre/post 融合（含 |diff| 与 乘积交互）-> 对 N 维输出 logits；value 用全局均值+info_f。
    """
    def __init__(self, d_info: int, d_cand: int, hidden: int):
        super().__init__()
        self.H = hidden
        # 1) info 编码（你原来就是展平 L 再 MLP）
        self.info_enc = nn.Sequential(
            nn.Linear(d_info * 2 * cfg.diff_pre_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        # 2) 候选编码：展平 (L2*Dc) -> H
        flat_dim = cfg.diff_infer_len * d_cand  # L2 * Dc
        self.cand_pre_proj  = nn.Sequential(nn.Linear(flat_dim, hidden), nn.ReLU())
        self.cand_post_proj = nn.Sequential(nn.Linear(flat_dim, hidden), nn.ReLU())

        # 3) FiLM 条件化：由 info_f 产生 (gamma,beta) 作用到候选嵌入
        self.film = nn.Sequential(
            nn.Linear(hidden, hidden*2)  # 输出 concat[gamma, beta]
        )

        # 4) 融合 + 分类头（在 N 上打分）
        self.pair_fuse = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.cls = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        # 5) 值函数（baseline）
        self.vf  = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(
        self,
        info_pre: torch.Tensor,   # (B, L, D)
        info_post: torch.Tensor,  # (B, L, D)
        cand_pre: torch.Tensor,   # (B, N, L2, Dc)
        cand_post: torch.Tensor   # (B, N, L2, Dc)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, L2, Dc = cand_pre.shape
        H = self.H

        # print(f'sjn :info{info_pre.shape} {info_post.shape} cand{cand_pre.shape} {cand_post.shape}')
        # print(f'sjn info_pre: {info_pre.shape}, info_post: {info_post.shape}, cand_pre: {cand_pre.shape}, cand_post: {cand_post.shape}')

        # ---- info 编码 ----
        info_p = info_pre.reshape(B, -1)            # (B, L*D)
        info_q = info_post.reshape(B, -1)           # (B, L*D)
        info_f = self.info_enc(torch.cat([info_p, info_q], dim=-1))  # (B, H)


        # ---- 候选展平 + 线性投影 ----
        pre_flat  = cand_pre.reshape(B, N, L2*Dc)                 # (B,N,L2*Dc)
        post_flat = cand_post.reshape(B, N, L2*Dc)                # (B,N,L2*Dc)
        pre_z  = self.cand_pre_proj(pre_flat)                     # (B,N,H)
        post_z = self.cand_post_proj(post_flat)                   # (B,N,H)

        # ---- FiLM 条件化（基于 info_f）----
        gamma_beta = self.film(info_f)                            # (B, 2H)
        gamma, beta = gamma_beta[:, :H], gamma_beta[:, H:]        # (B,H)
        # 扩展到 N 维
        gamma = gamma.unsqueeze(1)                                # (B,1,H)
        beta  = beta.unsqueeze(1)                                 # (B,1,H)
        pre_c  = gamma * pre_z  + beta                            # (B,N,H)
        post_c = gamma * post_z + beta                            # (B,N,H)

        # ---- 成对融合（对称 + 交互）----
        fuse = torch.cat([pre_c, post_c, torch.abs(pre_c - post_c), pre_c * post_c], dim=-1)  # (B,N,4H)
        pair_feat = self.pair_fuse(fuse)                                                               # (B,N,H)

        # ---- 在 N 上打分 ----
        logits = self.cls(pair_feat).squeeze(-1)                       # (B,N)

        # ---- 值函数（baseline）----
        cand_mean = pair_feat.mean(dim=1)                               # (B,H)
        vf_in = torch.cat([info_f, cand_mean, info_f * cand_mean], dim=-1)  # (B,3H)
        value = self.vf(vf_in).squeeze(-1)                              # (B,)
        # print(f'sjn :logits{logits.shape} value{value.shape}')
        return logits, value

class PairPolicy1(nn.Module):
    """基于(info_pre, info_post, (cand_pre, cand_post))输出候选logits与值函数"""
    def __init__(self, d_info: int, d_cand: int, hidden: int):
        super().__init__()
        self.info_encode = nn.Linear(d_info, hidden)
        self.info_enc = nn.Sequential(
            nn.Linear(d_info * 2 * cfg.diff_pre_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.cpool = None # SeqPool(d_cand, hidden)
        self.cls = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.vf  = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(
        self,
        info_pre: torch.Tensor,  # (B, L, D)
        info_post: torch.Tensor, # (B, L, D)
        cand_pre: torch.Tensor,  # (B, N, L2, D)
        cand_post: torch.Tensor  # (B, N, L2, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, L2, Dc = cand_pre.shape
        D = info_pre.size(-1)
        # enc info
        info_p = info_pre.view(B, -1)
        #info_p = info_pre.mean(dim=1)
        info_q = info_post.view(B,-1)
        # info_q = info_post.mean(dim=1)
        info_f = self.info_enc(torch.cat([info_p, info_q], dim=-1))  # (B, H)

        # enc pair candidates
        pre_f  = self.cpool(cand_pre.reshape(B * N, L2, Dc))   # (B*N, H)
        post_f = self.cpool(cand_post.reshape(B * N, L2, Dc))  # (B*N, H)
        pair_f = torch.cat([pre_f, post_f], dim=-1)            # (B*N, 2H)

        logits = self.cls(pair_f).reshape(B, N)                 # (B, N)

        # value：用 info_f + 候选总体统计（均值）融合
        cand_f_mean = pair_f.reshape(B, N, -1).mean(dim=1)      # (B, 2H)
        vf_in = torch.cat([info_f, cand_f_mean], dim=-1)        # (B, 3H)
        value = self.vf(vf_in).squeeze(-1)                      # (B,)
        return logits, value

# -------------------- sampling --------------------
def gumbel_topk(logits: torch.Tensor, k: int, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      idx: (B, k)
      logp_selected_sum: (B,) 选中集合的近似对数概率（将所选项的 log_softmax 概率求和）
    """
    if greedy:
        idx = torch.topk(logits, k, dim=1).indices
        logp = F.log_softmax(logits, dim=1).gather(1, idx).sum(dim=1)
        return idx, logp
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-12)))
    y = logits + g
    idx = y.topk(k, dim=1).indices
    logp = F.log_softmax(logits, dim=1).gather(1, idx).sum(dim=1)
    return idx, logp

def take_pairs(hydra_pre: torch.Tensor, hydra_post: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据(B,k)的idx，从(B,N,L2,D)取成(B,k,L2,D)；pre/post用相同索引，保证结对"""
    B, N, L2, D = hydra_pre.shape
    k = idx.size(1)
    ar = torch.arange(B, device=hydra_pre.device).unsqueeze(1).expand(B, k)
    sel_pre  = hydra_pre[ar, idx]   # (B,k,L2,D)
    sel_post = hydra_post[ar, idx]  # (B,k,L2,D)
    return sel_pre, sel_post

# -------------------- RL 算法封装 --------------------
class RLSelector(nn.Module):
    """
    统一接口：
      - choose(): 前向选择（train时采样，eval时贪心）
      - step():   单步训练（内部完成策略/值函数/熵正则/PPO剪切等）
    """
    def __init__(self, d_info: int, d_cand: int, cfg: RLConfig):
        super().__init__()
        self.cfg = cfg
        self.policy = PairPolicy(d_info, d_cand, cfg.hidden)

    @torch.no_grad()
    def choose(
        self,
        info_pre: torch.Tensor, info_post: torch.Tensor,
        hydra_pre: torch.Tensor, hydra_post: torch.Tensor,
        greedy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.policy(info_pre, info_post, hydra_pre, hydra_post)  # (B,N), (B,)
        idx, logp = gumbel_topk(logits, self.cfg.k_select, greedy=greedy)       # (B,k), (B,)
        sel_pre, sel_post = take_pairs(hydra_pre, hydra_post, idx)
        # out = {"logits": logits, "logp": logp, "value": value, "idx": idx}
        return sel_pre, sel_post

    def _entropy_selected(self, logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # 近似熵：仅用被选项的对数概率取负并取均值；也可改满分布熵
        sel_lp = F.log_softmax(logits, dim=1).gather(1, idx)  # (B,k)
        return -(sel_lp.mean(dim=1))                          # (B,)

    def _compute_adv(self, reward: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # 单步场景 advantage = r - V(s)
        return reward - value.detach()

    def step(
        self,
        info_pre: torch.Tensor, info_post: torch.Tensor,
        hydra_pre: torch.Tensor, hydra_post: torch.Tensor,
        match,                 # 回调: (info_pre, info_post, sel_pre, sel_post) -> (loss_per_sample: (B,))
        optim: torch.optim.Optimizer,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        完成一次"采样→下游评估→构造reward→RL更新"
        downstream_loss_fn 必须返回 (B,) 的样本级 loss（建议你的loss支持 reduction='none'）
        """
        cfg = self.cfg
        # 1) 采样（train模式）
        logits, value = self.policy(info_pre, info_post, hydra_pre, hydra_post)   # (B,N),(B,)
        idx, logp_old = gumbel_topk(logits, cfg.k_select, greedy=False)           # (B,k),(B,)
        sel_pre, sel_post = take_pairs(hydra_pre, hydra_post, idx)

        # 2) 下游计算 per-sample loss 与 reward
        loss_vec= downstream_fn(match, info_pre, info_post, sel_pre, sel_post, labels)     # (B,)
        # 标准化 reward（稳定训练）
        reward = -loss_vec.detach()
        mu, sd = reward.mean(), reward.std().clamp_min(1e-6)
        reward = (reward - mu) / sd
        # print('sjn', reward.shape, value.shape)
        # 3) RL 更新：bandit / a2c / ppo
        logs = {}


        if cfg.algo == "bandit":
            # REINFORCE + 熵正则
            entropy = self._entropy_selected(logits, idx)               # (B,)
            policy_loss = -(reward * logp_old).mean()
            ent_loss = - cfg.ent_coef * entropy.mean()
            loss = policy_loss + ent_loss
            optim.zero_grad()
            loss.backward()
            if cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            optim.step()
            logs.update(dict(rl_loss=float(loss.item()), pol=float(policy_loss.item()), vf=0.0))

        elif cfg.algo == "a2c":
            # 优势 + 值函数 + 熵
            entropy = self._entropy_selected(logits, idx)
            adv = self._compute_adv(reward, value)                      # (B,)
            policy_loss = -(adv * logp_old).mean()
            value_loss  = F.mse_loss(value, reward)
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy.mean()
            optim.zero_grad()
            loss.backward()
            if cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            optim.step()
            logs.update(dict(rl_loss=float(loss.item()), pol=float(policy_loss.item()), vf=float(value_loss.item())))

        else:  # "ppo"
            # 单步 PPO：使用当前batch作为老策略（detach），做K轮clip更新
            with torch.no_grad():
                old_logits = logits.clone()
                old_value  = value.clone()
                old_logp   = logp_old.clone()
                adv        = self._compute_adv(reward, old_value)
                # 归一化 advantage（更稳）
                a_m, a_s = adv.mean(), adv.std().clamp_min(1e-6)
                adv = (adv - a_m) / a_s

            # 多epoch（单minibatch）
            pol_loss_acc = 0.0
            vf_loss_acc  = 0.0
            for _ in range(cfg.ppo_epochs):
                logits_new, value_new = self.policy(info_pre, info_post, hydra_pre, hydra_post)
                # 重新用同一 idx 近似新策略 logp（简化，无需重采样）
                logp_new = F.log_softmax(logits_new, dim=1).gather(1, idx).sum(dim=1)  # (B,)
                ratio = torch.exp(logp_new - old_logp)                                  # (B,)
                surr1 =  ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                policy_loss = -(torch.min(surr1, surr2)).mean()
                value_loss  = F.mse_loss(value_new, reward)
                entropy     = self._entropy_selected(logits_new, idx).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                optim.zero_grad()
                loss.backward()
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                optim.step()

                pol_loss_acc += float(policy_loss.item())
                vf_loss_acc  += float(value_loss.item())

            logs.update(dict(
                rl_loss=pol_loss_acc + vf_loss_acc,
                pol=pol_loss_acc / cfg.ppo_epochs,
                vf=vf_loss_acc  / cfg.ppo_epochs
            ))

        # 4) 返回训练日志与选中的对（可用于可视化/对比）
        logs["reward_mean"] = float(reward.mean().item())
        logs["down_loss_mean"] = float(loss_vec.mean().item())
        return logs
# === end of rl_selector.py ====================================================