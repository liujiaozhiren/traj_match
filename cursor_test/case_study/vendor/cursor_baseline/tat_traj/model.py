# -*- coding: utf-8 -*-
"""
Tracklet Association Tracker (TAT) — **轨迹 pre/post 上的讨论用 baseline 骨架**

参考论文语境：*Tracklet Association Tracker: An End-to-End Learning-based Association
Approach for Multi-Object Tracking* (arXiv:1808.01562 等). 原设定是 **多检测 → 短 tracklet
→ tracklet 图上全局/层级关联**；在你仓库的数据里 **每条样本只有 pre、post 两段**，
因此「多节点 tracklet 图」**退化为 2 个节点（或 batch 内 B 个样本 × 每样本 2 段）**，
TAT 相对 DAN 的 **结构优势主要体现不出来**，差异会更多落在 **关联目标 / loss /
是否显式建图消息传递** 上 —— 本文件用注释标出后续可改点。

---------------------------------------------------------------------------
**本 baseline 当前阶段（phase 0，可跑、可与 DAN 横比 valid）**
---------------------------------------------------------------------------

- **输入**：与 ``train_dan_traj`` / ``pair_pre_post_lonlat12`` 相同，``(B, L, 2)`` lon/lat。
- **两段语义**：``pre``、``post`` 视为 **两个 tracklet 节点**；各用同一套 ``TrackletEncoder``
  得到节点特征 ``h_pre, h_post``。
- **可选** ``node_update``：占位，模拟「图上做一轮 message passing」；默认 ``nn.Identity``，
  置 ``use_graph_update=True`` 时用轻量自注意力在 **(h_pre, h_post)** 两节点上更新（仅作
  讨论占位，**不是** TAT 原文完整图模块）。
- **batch 内匹配头**：与简化 DAN 一样，对所有 ``(i, j)`` 拼接 ``[h_pre[i], h_post[j]]`` → MLP
  → ``(B, B)`` logits，训练 **行 CE**、验证 **k×k + retrieval_metrics** —— 这样 **valid
  协议与 ``train_repr_baseline`` / ``train_dan_traj`` 严格一致**，便于你对比；**不是**
  TAT 原文的 bi-level 优化，后续可在此处换 loss / 换头。

---------------------------------------------------------------------------
**实现讨论清单（按优先级）**
---------------------------------------------------------------------------

1. **Loss**：行 CE → TAT / 匹配文献中的 **结构化目标**（例如匈牙利对齐、pairwise margin、
   或 batch 内最优传输正则）；若只做 pre/post 二段，是否与 **对角 CE** 等价类要单独推。
2. **图**：若未来一条样本有多段（>2 tracklet），在此扩展 ``node_update`` 与邻接矩阵；
   当前 2 节点时 GNN 与 MLP 边界需实验界定。
3. **Unmatched / 背景类**：与 DAN-full 的 ``(B+1)²`` 类似，是否在 **只有两段** 时仍需要
   dummy 维（讨论点）。
4. **推理**：当前与 DAN 相同输出 ``(B,B)``；TAT 式「直接输出关联指派」需改 metrics 侧接口。

conda: ``traj_match``。训练入口：``cursor_baseline/train_tat_traj.py``。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrackletEncoder(nn.Module):
    """单段轨迹 → 定长向量（与 DAN 的 TrajFeatureNet 同构，命名强调 tracklet）。"""

    def __init__(self, *, rnn_hidden: int = 128, embed_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.rnn_hidden = int(rnn_hidden)
        self.embed_dim = int(embed_dim)
        self.rnn = nn.GRU(
            input_size=2,
            hidden_size=self.rnn_hidden,
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
        )
        rnn_out = 2 * self.rnn_hidden
        self.out_proj = nn.Linear(rnn_out, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.out_proj(y.mean(dim=1))


class _TwoNodeGraphUpdate(nn.Module):
    """
    (B, 2, C) 上的一层 self-attention，仅作「tracklet 图消息传递」占位；节点顺序 [pre, post]。
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(nodes, nodes, nodes, need_weights=False)
        return self.norm(nodes + out)


class TATTrajTwoSegmentBaseline(nn.Module):
    """
    pre/post 两段 → 节点特征 → 可选 2 节点图更新 → batch 内 ``(B,B)`` 关联 logits。

    ``forward(pre, post)`` 与 ``DANMotTrajSimplified`` 同签名，便于复用 ``dan_valid_metrics``。
    """

    def __init__(
        self,
        *,
        rnn_hidden: int = 128,
        embed_dim: int = 128,
        rnn_layers: int = 1,
        mlp_hidden: int = 256,
        use_graph_update: bool = False,
    ):
        super().__init__()
        self.encoder_pre = TrackletEncoder(rnn_hidden=rnn_hidden, embed_dim=embed_dim, num_layers=rnn_layers)
        self.encoder_post = TrackletEncoder(rnn_hidden=rnn_hidden, embed_dim=embed_dim, num_layers=rnn_layers)
        self.use_graph_update = bool(use_graph_update)
        self.node_update: nn.Module = _TwoNodeGraphUpdate(embed_dim) if self.use_graph_update else nn.Identity()
        in_pair = 2 * embed_dim
        self.pair_ln = nn.LayerNorm(in_pair)
        self.assoc_mlp = nn.Sequential(
            nn.Linear(in_pair, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        h_pre = self.encoder_pre(pre)
        h_post = self.encoder_post(post)
        if self.use_graph_update:
            # (B, 2, C): 两个 tracklet 节点
            nodes = torch.stack([h_pre, h_post], dim=1)
            nodes = self.node_update(nodes)
            h_pre, h_post = nodes[:, 0], nodes[:, 1]
        b, c = h_pre.shape
        pre_e = h_pre.unsqueeze(1).expand(b, b, c)
        post_e = h_post.unsqueeze(0).expand(b, b, c)
        pair = torch.cat([pre_e, post_e], dim=-1)
        z = self.pair_ln(pair.reshape(b * b, -1))
        return self.assoc_mlp(z).view(b, b)
