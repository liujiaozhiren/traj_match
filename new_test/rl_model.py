import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F

from match_model.my.my_infer_demo import pair_loss


class HydraContextEncoder(nn.Module):
    def __init__(self, d_info: int, d_cand: int, hidden: int):
        super().__init__()
        self.H = hidden
        self.info_enc = nn.Sequential(
            nn.Linear(d_info * 2 * cfg.diff_pre_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        flat_dim = cfg.diff_infer_len * d_cand  # L2 * Dc
        self.cand_pre_proj = nn.Sequential(nn.Linear(flat_dim, hidden), nn.ReLU())
        self.cand_post_proj = nn.Sequential(nn.Linear(flat_dim, hidden), nn.ReLU())
        self.film = nn.Sequential(
            nn.Linear(hidden, hidden * 4)  # 输出 concat[gamma, beta]
        )

        self.cls_pre = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.cls_post = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, info_pre, info_post, cand_pre, cand_post):
        B, N, L2, Dc = cand_pre.shape
        H = self.H

        # ---- info 编码 ----
        info_p = info_pre.reshape(B, -1)  # (B, L*D)
        info_q = info_post.reshape(B, -1)  # (B, L*D)
        info_f = self.info_enc(torch.cat([info_p, info_q], dim=-1))  # (B, H)

        # ---- 候选展平 + 线性投影 ----
        pre_flat = cand_pre.reshape(B, N, L2 * Dc)  # (B,N,L2*Dc)
        post_flat = cand_post.reshape(B, N, L2 * Dc)  # (B,N,L2*Dc)
        pre_z = self.cand_pre_proj(pre_flat)  # (B,N,H)
        post_z = self.cand_post_proj(post_flat)  # (B,N,H)

        # ---- info_f 和 pre_z post_z 合并----
        gamma_beta = self.film(info_f)  # (B, 4H)
        gamma_pre, gamma_post = gamma_beta[:, :H], gamma_beta[:, H:2*H]  # (B,H)
        beta_pre, beta_post = gamma_beta[:, 2*H:3*H], gamma_beta[:, 3*H:4*H]
        # 扩展到 N 维
        gamma_pre = gamma_pre.unsqueeze(1)  # (B,1,H)
        gamma_post = gamma_post.unsqueeze(1)  # (B,1,H)
        beta_pre = beta_pre.unsqueeze(1)
        beta_post = beta_post.unsqueeze(1)

        pre_c = gamma_pre * pre_z + beta_pre    # (B,N,H)
        post_c = gamma_post * post_z + beta_post    # (B,N,H)

        return pre_c, post_c


class Encoder2(nn.Module):
    def __init__(self, d_info, d_cand, d_hidden):
        super().__init__()
        self.encoder = None
        self.net_pre = nn.Sequential(
            nn.Linear(cfg.match_dim*2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU()
        )
        self.net_post = nn.Sequential(
            nn.Linear(cfg.match_dim*2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU()
        )
    def register(self, encoder):
        self.encoder = encoder

    def forward(self, info_pre, info_post, cand_pre, cand_post):
        # x: (..., d_in) -> (..., d_out)
        B, n, d = info_pre.shape[0], cand_pre.shape[1], cand_pre.shape[-1]
        pre_c, post_c = self.encoder.encode(info_pre, info_post, cand_pre, cand_post)
        ret = self.net_pre(pre_c.view(B,n,-1)), self.net_post(post_c.view(B,n,-1))
        return ret


class Encoder(nn.Module):
    """简单表征: x -> z"""
    def __init__(self, d_info, d_cand, d_hidden):
        super().__init__()
        self.encoder = HydraContextEncoder(d_info=d_info, d_cand=d_cand, hidden=d_hidden)
        self.net_pre = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU()
        )
        self.net_post = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU()
        )

    def forward(self, info_pre, info_post, cand_pre, cand_post):
        # x: (..., d_in) -> (..., d_out)
        pre_c, post_c = self.encoder(info_pre, info_post, cand_pre, cand_post)
        ret = self.net_pre(pre_c), self.net_post(post_c)
        return ret

    # def super_encode(self, match, info_pre, info_post, cand_pre, cand_post):
    #     B, n, d = info_pre.shape[0], cand_pre.shape[1], cand_pre.shape[-1]
    #     pre_c, post_c = match.encode(info_pre, info_post, cand_pre, cand_post)
    #     ret = self.net_pre(pre_c.view()), self.net_post(post_c)

#
# class DeepLinUCB:
#     def __init__(
#         self,
#         num,
#         d_context: int,
#         rep_dim: int = 32,
#         alpha: float = 1.0,
#         lambda_: float = 1.0,
#         device: str = None,
#     ):
#         self.num=num
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.alpha = alpha
#         self.lambda_ = lambda_
#
#         self.encoder = Encoder(d_info=d_context, d_cand=d_context, d_hidden=rep_dim).to(self.device)
#         self.rep_dim = rep_dim
#         self.A_pre = lambda_ * torch.eye(rep_dim, device=self.device)
#         self.A_post = lambda_ * torch.eye(rep_dim, device=self.device)
#         self.A_inv_pre = torch.inverse(self.A_pre)
#         self.A_inv_post = torch.inverse(self.A_post)
#         self.b_pre = torch.zeros(rep_dim, device=self.device)
#         self.b_post = torch.zeros(rep_dim, device=self.device)
#
#         self.opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
#         self.ctx_cache= None
#
#     @torch.no_grad()
#     def choose_step(self, info_pre, info_post, hydra_pre, hydra_post):
#         self.encoder.eval()
#         Z_pre, Z_post = self.encoder(info_pre, info_post, hydra_pre, hydra_post)
#         chosen_idx_pre = []
#         chosen_idx_post = []
#         B = info_pre.size(0)
#         for b in range(B):
#             Zb_pre = Z_pre[b]  # (K, H)
#             Zb_post = Z_post[b]  # (K, H)
#             idx_pre_b, idx_post_b = self._select_topk(Zb_pre, Zb_post, self.num)
#             chosen_idx_pre.append(idx_pre_b)
#             chosen_idx_post.append(idx_post_b)
#         chosen_idx_pre = torch.stack(chosen_idx_pre, dim=0)  # (B, num)
#         chosen_idx_post = torch.stack(chosen_idx_post, dim=0)  # (B, num)
#         hydra_pre_sel = gather_by_index(hydra_pre, chosen_idx_pre)  # (B, num, L2, Dc)
#         hydra_post_sel = gather_by_index(hydra_post, chosen_idx_post)  # (B, num, L2, Dc)
#         self.ctx_cache = info_pre, info_post, hydra_pre, hydra_post
#         return hydra_pre_sel, hydra_post_sel
#
#     def _select_topk(self, Z_pre, Z_post, k):
#         theta_hat_pre = self.A_inv_pre @ self.b_pre  # (rep_dim,)
#         mu_pre = Z_pre @ theta_hat_pre  # (K,)
#
#         Az_pre = Z_pre @ self.A_inv_pre  # (K, rep_dim)
#         sigma_sq_pre = (Az_pre * Z_pre).sum(dim=1)  # (K,)
#         sigma_pre = torch.sqrt(torch.clamp(sigma_sq_pre, min=1e-8))
#
#         ucb_pre = mu_pre + self.alpha * sigma_pre  # (K,)
#         topk_pre = torch.topk(ucb_pre, k=min(k, Z_pre.size(0))).indices  # (k,)
#
#         theta_hat_post = self.A_inv_post @ self.b_post  # (rep_dim,)
#         mu_post = Z_post @ theta_hat_post
#         Az_post = Z_post @ self.A_inv_post
#         sigma_sq_post = (Az_post * Z_post).sum(dim=1)
#         sigma_post = torch.sqrt(torch.clamp(sigma_sq_post, min=1e-8))
#         ucb_post = mu_post + self.alpha * sigma_post
#         topk_post = torch.topk(ucb_post, k=min(k, Z_post.size(0))).indices
#         return topk_pre, topk_post
#
#
#     def update_batch(self, r_sel: torch.Tensor):
#         """
#         X_sel: (N, d_context)   N = B * prefix，所有被选中的 hydra 的特征堆在一起
#         r_sel: (N,)             对应每个选择的 reward（可以 B 的 reward repeat_interleave prefix）
#         """
#         self.encoder.train()
#         info_pre, info_post, hydra_pre, hydra_post = self.ctx_cache
#
#         # 1) 更新 encoder：让 theta_hat^T z(x) 逼近 reward
#         Z_pre, Z_post = self.encoder(info_pre, info_post, hydra_pre, hydra_post)
#
#         with torch.no_grad():
#             theta_hat_pre = self.A_inv_pre @ self.b_pre  # (rep_dim,)
#             theta_hat_post = self.A_inv_post @ self.b_post
#         r_sel = r_sel.view()
#         pred_pre = Z_pre @ theta_hat_pre             # (N,)
#         loss_pre = F.mse_loss(pred_pre, r_sel)
#         pred_post = Z_post @ theta_hat_post           # (N,)
#         loss_post = F.mse_loss(pred_post, r_sel)
#         loss = loss_pre + loss_post
#
#         self.opt.zero_grad()
#         loss.backward()
#         self.opt.step()
#
#         # 2) 更新线性头的统计量
#         with torch.no_grad():
#             Z_pre, Z_post = self.encoder(info_pre, info_post, hydra_pre, hydra_post)
#             for i in range(Z_pre.size(0)):
#                 z_pre = Z_pre[i].detach().view(-1, 1)  # (rep_dim, 1)
#                 self.A_pre += z_pre @ z_pre.T
#                 self.b_pre += r_sel[i] * z_pre.squeeze(1)
#                 z_post = Z_post[i].detach().view(-1, 1)
#                 self.A_post += z_post @ z_post.T
#                 self.b_post += r_sel[i] * z_post.squeeze(1)
#             self.A_inv_pre = torch.inverse(self.A_pre)
#             self.A_inv_post = torch.inverse(self.A_post)


class DeepLinUCB:
    def __init__(
        self,
        num,
        d_context: int,
        rep_dim: int = 32,
        alpha: float = 1.0,
        lambda_: float = 1.0,
        device: str = None,
    ):
        self.num = num
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.lambda_ = lambda_

        # d_context 在你的实现里其实用的是 d_info / d_cand，逻辑 OK
        self.encoder = Encoder(d_info=d_context, d_cand=d_context, d_hidden=rep_dim).to(self.device)

        self.rep_dim = rep_dim
        self.A_pre = lambda_ * torch.eye(rep_dim, device=self.device)
        self.A_post = lambda_ * torch.eye(rep_dim, device=self.device)
        self.A_inv_pre = torch.inverse(self.A_pre)
        self.A_inv_post = torch.inverse(self.A_post)
        self.b_pre = torch.zeros(rep_dim, device=self.device)
        self.b_post = torch.zeros(rep_dim, device=self.device)

        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)

        # 存上一轮用于 update 的 context 和 index
        self.ctx_cache = None

    def freeze(self, flag=True):
        for param in self.encoder.parameters():
            param.requires_grad = not flag

    @torch.no_grad()
    def choose_step(self, info_pre, info_post, hydra_pre, hydra_post):
        """
        输入:
          info_*:  (B, ...)
          hydra_*: (B, N, L2, Dc)
        输出:
          hydra_*_sel: (B, num, L2, Dc)
        同时在 ctx_cache 里缓存 (info_*, hydra_*, idx_*)
        """
        self.encoder.eval()

        info_pre = info_pre.to(self.device)
        info_post = info_post.to(self.device)
        hydra_pre = hydra_pre.to(self.device)
        hydra_post = hydra_post.to(self.device)

        Z_pre, Z_post = self.encoder(info_pre, info_post, hydra_pre, hydra_post)  # (B,N,rep_dim)

        B, N, H = Z_pre.shape
        chosen_idx_pre = []
        chosen_idx_post = []

        for b in range(B):
            Zb_pre = Z_pre[b]   # (N, H)
            Zb_post = Z_post[b] # (N, H)
            idx_pre_b, idx_post_b = self._select_topk(Zb_pre, Zb_post, self.num)
            chosen_idx_pre.append(idx_pre_b)
            chosen_idx_post.append(idx_post_b)

        chosen_idx_pre = torch.stack(chosen_idx_pre, dim=0)   # (B, num)
        chosen_idx_post = torch.stack(chosen_idx_post, dim=0) # (B, num)

        hydra_pre_sel = gather_by_index(hydra_pre, chosen_idx_pre)   # (B,num,L2,Dc)
        hydra_post_sel = gather_by_index(hydra_post, chosen_idx_post)

        # 缓存本轮的 context 和 index，update_batch 会用
        self.ctx_cache = (
            info_pre.detach(),
            info_post.detach(),
            hydra_pre.detach(),
            hydra_post.detach(),
            chosen_idx_pre.detach(),
            chosen_idx_post.detach(),
        )

        return hydra_pre_sel, hydra_post_sel

    def _select_topk(self, Z_pre, Z_post, k):
        """
        Z_pre/Z_post: (N, rep_dim)
        """
        theta_hat_pre = self.A_inv_pre @ self.b_pre  # (rep_dim,)
        mu_pre = Z_pre @ theta_hat_pre               # (N,)
        Az_pre = Z_pre @ self.A_inv_pre              # (N, rep_dim)
        sigma_sq_pre = (Az_pre * Z_pre).sum(dim=1)   # (N,)
        sigma_pre = torch.sqrt(torch.clamp(sigma_sq_pre, min=1e-8))
        ucb_pre = mu_pre + self.alpha * sigma_pre
        topk_pre = torch.topk(ucb_pre, k=min(k, Z_pre.size(0))).indices  # (k,)

        theta_hat_post = self.A_inv_post @ self.b_post
        mu_post = Z_post @ theta_hat_post
        Az_post = Z_post @ self.A_inv_post
        sigma_sq_post = (Az_post * Z_post).sum(dim=1)
        sigma_post = torch.sqrt(torch.clamp(sigma_sq_post, min=1e-8))
        ucb_post = mu_post + self.alpha * sigma_post
        topk_post = torch.topk(ucb_post, k=min(k, Z_post.size(0))).indices

        return topk_pre, topk_post

    def update_batch(self, r_sel: torch.Tensor):
        """
        r_sel: (B * num,)  reward_pos/neg.repeat_interleave(num)
        使用: 调用顺序必须是 choose_step -> update_batch
        """
        assert self.ctx_cache is not None, "update_batch called before choose_step"

        (
            info_pre,
            info_post,
            hydra_pre,
            hydra_post,
            idx_pre,
            idx_post,
        ) = self.ctx_cache

        self.encoder.train()
        info_pre = info_pre.to(self.device)
        info_post = info_post.to(self.device)
        hydra_pre = hydra_pre.to(self.device)
        hydra_post = hydra_post.to(self.device)
        idx_pre = idx_pre.to(self.device)
        idx_post = idx_post.to(self.device)
        r_sel = r_sel.to(self.device)

        B = info_pre.size(0)
        k = idx_pre.size(1)

        # 重新编码得到 Z_pre/Z_post: (B,N,H)
        Z_pre, Z_post = self.encoder(info_pre, info_post, hydra_pre, hydra_post)

        # 只取被选中的 top-k 位置 -> (B*k, H)
        Z_sel_pre_list = []
        Z_sel_post_list = []
        for b in range(B):
            Z_sel_pre_list.append(Z_pre[b, idx_pre[b]])   # (k,H)
            Z_sel_post_list.append(Z_post[b, idx_post[b]])
        Z_sel_pre = torch.cat(Z_sel_pre_list, dim=0)      # (B*k, H)
        Z_sel_post = torch.cat(Z_sel_post_list, dim=0)    # (B*k, H)

        assert r_sel.numel() == Z_sel_pre.size(0), \
            f"r_sel.size({r_sel.numel()}) vs Z_sel_pre.size(0)={Z_sel_pre.size(0)} 不一致"

        # -------- 1) 用线性头的 theta_hat 拟合 reward，更新 encoder --------
        with torch.no_grad():
            theta_hat_pre = self.A_inv_pre @ self.b_pre     # (H,)
            theta_hat_post = self.A_inv_post @ self.b_post  # (H,)

        pred_pre = Z_sel_pre @ theta_hat_pre    # (B*k,)
        pred_post = Z_sel_post @ theta_hat_post

        loss_pre = F.mse_loss(pred_pre, r_sel)
        loss_post = F.mse_loss(pred_post, r_sel)
        loss = loss_pre + loss_post

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # -------- 2) 用 Z_sel 和 r_sel 更新线性 UCB 统计量 --------
        with torch.no_grad():
            # 用更新后的 encoder 再算一遍 Z_sel（更干净）
            Z_pre_new, Z_post_new = self.encoder(info_pre, info_post, hydra_pre, hydra_post)
            Z_sel_pre_list = []
            Z_sel_post_list = []
            for b in range(B):
                Z_sel_pre_list.append(Z_pre_new[b, idx_pre[b]])
                Z_sel_post_list.append(Z_post_new[b, idx_post[b]])
            Z_sel_pre = torch.cat(Z_sel_pre_list, dim=0)   # (B*k,H)
            Z_sel_post = torch.cat(Z_sel_post_list, dim=0) # (B*k,H)

            for i in range(Z_sel_pre.size(0)):
                z_pre = Z_sel_pre[i].view(-1, 1)   # (H,1)
                self.A_pre += z_pre @ z_pre.T
                self.b_pre += r_sel[i] * z_pre.squeeze(1)

                z_post = Z_sel_post[i].view(-1, 1)
                self.A_post += z_post @ z_post.T
                self.b_post += r_sel[i] * z_post.squeeze(1)

            self.A_inv_pre = torch.inverse(self.A_pre)
            self.A_inv_post = torch.inverse(self.A_post)

def gather_by_index(hydra, idx_mat):
    B = hydra.shape[0]
    k = idx_mat.shape[1]
    idx = idx_mat.unsqueeze(-1)  # (B, K, 1)
    expand_shape = [B, k] + [1] * (hydra.dim() - 2)
    idx = idx.view(*expand_shape).expand(-1, -1, *hydra.shape[2:])
    selected = torch.gather(hydra, dim=1, index=idx)
    return selected

def get_reward_by_each(match, base_loss, label, info_pre, info_post, hydra_pre_base, hydra_post_base, hydra_pre_sel, hydra_post_sel):
    # pos
    B, n = hydra_pre_base.size(0), hydra_pre_base.size(1)
    # hydra_pre_sel, hydra_post_sel = bandit.choose_step(info_pre, info_post, hydra_pre, hydra_post)
    reward_dict = torch.zeros((B, n,n), device=base_loss.device)
    for i in range(n):
        # reward_list = tor.
        for j in range(n):
            use_hydra_pre = hydra_pre_base.clone()
            use_hydra_post = hydra_post_base.clone()
            use_hydra_pre[:, j] = hydra_pre_sel[:, i]
            use_hydra_post[:, j] = hydra_post_sel[:, i]
            ret = match(info_pre, info_post, use_hydra_pre, use_hydra_post)
            loss_each = pair_loss(ret, label)
            reward_pos = base_loss.detach() - loss_each.detach()
            reward_dict[:, i, j] = reward_pos
    return reward_dict.mean(dim=2).view(-1)  # (B,n)


def get_reward_by_simple(match, base_loss_each, label, info_pre, info_post, hydra_pre_base, hydra_post_base, hydra_pre_sel, hydra_post_sel):
    # pos
    # 整组 RL 组合
    ret_sel = match(info_pre, info_post, hydra_pre_sel, hydra_post_sel)
    loss_sel_each = pair_loss(ret_sel, label)  # (B,)

    adv_each = (base_loss_each - loss_sel_each)  # (B,)
    return adv_each.repeat_interleave(hydra_pre_sel.size(1))  # (B*prefix,)