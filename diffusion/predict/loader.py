import torch
from torch.utils.data import DataLoader

import cfg
from diffusion.predict.pretrain import pretrain_lstm_pred
from model.handler import ModelHandler
import numpy as np
from typing import Sequence


def complete_ap(a: Sequence[float | int],
                L: int,
                base: int = 200):
    a = np.asarray(a, dtype=float)
    if a.size < 2:
        raise ValueError("前缀 a 至少需 2 个点才能估计公差")
    if L < len(a):
        raise ValueError("目标长度 L 不能小于前缀长度")

    # ---------- 1) 估计公差 d ----------
    d_raw = np.median(np.diff(a))             # 对噪声鲁棒
    d     = np.round(d_raw / base) * base     # 四舍五入到 base 的倍数
    if d == 0:
        d = base                              # 极端退化情形

    # ---------- 2) 估计首项 s ----------
    idxs = np.arange(len(a))
    s_candidates = a - idxs * d               # 理想应当都≈首项
    s_med = np.median(s_candidates)           # 抗离群
    s     = np.round(s_med / base) * base     # 约束为 base 倍数

    # ---------- 3) 生成完整序列 ----------
    b = s + np.arange(L) * d
    return b.astype(np.int64)


def complete_timestamp(max_l, traj):
    T = [point[0] for point in traj]
    T = complete_ap(T, max_l, base=200)
    return T


def loader_collate_fn(batch):
    batch_size = len(batch)
    pre_traj, post_traj = [],[]
    for i in range(batch_size):
        pre_traj.append(batch[i][1])
        post_traj.append(batch[i][2])
    # traj_s = []
    # traj_s.extend(pre_traj)
    # traj_s.extend(post_traj)
    d = len(pre_traj[0][0])
    lengths_pre = torch.tensor([len(traj) for traj in pre_traj], dtype=torch.int32, device=cfg.device)
    lengths_post = torch.tensor([len(traj) for traj in post_traj], dtype=torch.int32, device=cfg.device)
    max_len_pre, max_len_post = int(lengths_pre.max()), int(lengths_post.max())
    max_l = max(max_len_pre, max_len_post) + cfg.max_predict
    padded_pre = torch.zeros(len(pre_traj), max_l, d, dtype=torch.float64, device=cfg.device)
    padded_post = torch.zeros(len(post_traj), max_l, d, dtype=torch.float64, device=cfg.device)
    for i, traj in enumerate(pre_traj):
        L = len(traj)
        padded_pre[i, :L, :] = torch.tensor(traj, dtype=torch.float64, device=cfg.device)
        padded_pre[i, :, 0] = torch.tensor(complete_timestamp(max_l, traj), dtype=torch.float64, device=cfg.device)
    for i, traj in enumerate(post_traj):
        L = len(traj)
        padded_post[i, :L, :] = torch.tensor(traj[::-1], dtype=torch.float64, device=cfg.device)
        padded_post[i, :, 0] = torch.tensor(complete_timestamp(max_l, traj[::-1]), dtype=torch.float64, device=cfg.device)
    ret = ((padded_pre, padded_post), (lengths_pre, lengths_post)), None
    return ret

def pred_lstm_preproc(model_name, train_set, valid_set, db, all):
    #model = ModelHandler(cfg.dim, cfg.model_name).to(cfg.device)
    dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=loader_collate_fn)
    validloader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=loader_collate_fn)
    testloader = DataLoader(db, batch_size=cfg.batch_size, shuffle=False, collate_fn=loader_collate_fn)
    model = ModelHandler(dim=cfg.dim, model_name=model_name).to(cfg.device)
    model = pretrain_lstm_pred(model)
    return model, dataloader, validloader, testloader