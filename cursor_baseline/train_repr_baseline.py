"""
Canonical entry for representation baselines on noisy 12-point (lon,lat) pre/post.

`--encoder`: **traj2simvec** | **t3s** | **trajgat** | **neutraj** | **trajcl** | **st2vec** (repo `model/ST2Vec` ST_Encoder + graph from train trajs; see `st2vec_twin.py`).

- Loss: `pair_loss` + BCE for non-trajcl encoders; **trajcl** uses upstream **MoCo** (post-only two views).
- Negatives: same derangement on post batch (cycle shift) as `pipeline/load.get_hydra_graph_label`.
- Valid: same protocol as cursor_mod_4 ft_match (k≈15% valid × 3 passes); scores (k×k) are **−L2** embedding
  distance (higher = closer), acc/HR/CE.

No hydra / no cache — representation baseline only.

**Single entry:** run this file only (``cursor_baseline/train_repr_baseline.py``); select backbone with ``--encoder``.

Example (from repo root; set ``CUDA_VISIBLE_DEVICES``, conda ``python``, then flags)::

    CUDA_VISIBLE_DEVICES=9 /home/haitaoyuan/anaconda3/envs/traj_match/bin/python cursor_baseline/train_repr_baseline.py --encoder neutraj --output-dir cursor_baseline/results/neutraj_run1 --rnn-dim 128 --neutraj-recurrent LSTM --neutraj-grid-h 500 --neutraj-grid-w 500 --train-batch 16 --lr 1e-4 --device cuda:0

    CUDA_VISIBLE_DEVICES=9 /home/haitaoyuan/anaconda3/envs/traj_match/bin/python cursor_baseline/train_repr_baseline.py --encoder trajcl --output-dir cursor_baseline/results/trajcl_run1 --embed-dim 256 --train-batch 16 --lr 1e-4 --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.modules["cfg"] = importlib.import_module("cursor_baseline.model_lib.cfg")
import cfg  # noqa: E402

from cursor_baseline.lonlat_traj import pair_pre_post_lonlat12  # noqa: E402
from cursor_baseline.metrics import retrieval_metrics  # noqa: E402
from cursor_baseline.t3s_twin import T3STwin  # noqa: E402
from cursor_baseline.traj2simvec_twin import Traj2SimVecTwin  # noqa: E402
from cursor_baseline.neutraj_twin import NeuTrajTwin, lonlat_mean_std_from_pairs  # noqa: E402
from cursor_baseline.trajcl_official import (  # noqa: E402
    TrajCLOfficialBaseline,
    apply_trajcl_config,
    build_cellspace_from_lonlat_bounds,
    lonlat_bounds_from_train_pairs,
)
from cursor_baseline.st2vec_build import (  # noqa: E402
    batch_pairs_to_st2vec_inputs,
    build_st2vec_road_network,
    st2vec_valid_encode_batch,
)
from cursor_baseline.st2vec_twin import ST2VecTwin  # noqa: E402
from cursor_baseline.trajgat_twin import TrajGATTwin  # noqa: E402
from cursor_baseline.trajgat_build import (  # noqa: E402
    batch_pairs_to_graphs,
    build_trajgat_index,
    graphs_for_valid_indices,
)
from cursor_baseline.model_lib.losses import pair_loss  # noqa: E402
from cursor_baseline.model_lib.proc_pairs import clean_pairs, proc_multi_single_traj  # noqa: E402
from cursor_baseline.model_lib.sample_db import sample_db  # noqa: E402


def neg_l2_embedding_logits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a, b: (B, D) trajectory embeddings.
    Returns (B,) logits = negative Euclidean distance so larger means more similar (for BCEWithLogits positives).
    """
    return -(a - b).norm(p=2, dim=-1)


def derangement_indices(n: int, device: torch.device) -> torch.Tensor:
    """Same as pipeline/load.get_hydra_graph_label (cycle shift)."""
    if n == 1:
        raise ValueError("batch size=1 cannot build derangement negative; use drop_last or batch_size>=2")
    idx = torch.arange(n, device=device, dtype=torch.long)
    shift = torch.randint(1, n, (1,), device=device).item()
    return (idx + shift) % n


def _load_pairs(sample_ratio: float, sample_seed: int):
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    all_pairs = clean_pairs(mul_p + sing_p)
    valid = sample_db(all_pairs, ratio=sample_ratio, seed=sample_seed)
    train = [p for p in all_pairs if p not in valid]
    return train, valid


class PairListDataset(Dataset):
    def __init__(self, pairs: list):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        return self.pairs[i]


def collate_lonlat_noisy(batch: list, n_points: int, noise_std_deg: float, noise_base: int, device_cpu: torch.device):
    """Stack (B, L, 2) lon/lat on CPU."""
    pres: list[torch.Tensor] = []
    posts: list[torch.Tensor] = []
    for j, item in enumerate(batch):
        pre, post = pair_pre_post_lonlat12(
            item,
            n_points=n_points,
            noise_std_deg=noise_std_deg,
            noise_seed_pre=int(noise_base) + j * 3,
            noise_seed_post=int(noise_base) + j * 3 + 1,
            device=device_cpu,
        )
        pres.append(pre)
        posts.append(post)
    return torch.stack(pres, dim=0), torch.stack(posts, dim=0)


@dataclass
class TrainResult:
    best_acc: float
    best_hr5: float
    best_hr10: float
    best_hr20: float
    best_r5_at_10: float
    best_r10_at_20: float
    best_ckpt: str | None


def run_repr_baseline(
    *,
    output_dir: str,
    train_batch: int,
    valid_batch: int,
    lr: float,
    max_epochs: int,
    early_stop: int,
    sample_ratio: float,
    sample_seed: int,
    encoder: str,
    rnn_dim: int,
    embed_dim: int,
    t3s_lstm_hidden: int,
    t3s_tf_layers: int,
    t3s_nhead: int,
    trajgat_d_model: int = 128,
    trajgat_max_items: int = 50,
    trajgat_max_depth: int = 50,
    trajgat_num_layers: int = 3,
    trajgat_nhead: int = 8,
    trajgat_d_lap: int = 8,
    trajgat_d_input: int = 4,
    neutraj_grid_h: int = 500,
    neutraj_grid_w: int = 500,
    neutraj_recurrent: str = "LSTM",
    trajcl_cell_size_m: float = 100.0,
    trajcl_cellspace_buffer_m: float = 500.0,
    trajcl_moco_nqueue: int = 2048,
    trajcl_moco_proj_dim: int = 128,
    trajcl_moco_temperature: float = 0.05,
    trajcl_aug1: str = "mask",
    trajcl_aug2: str = "subset",
    trajcl_bounds_max_pairs: int = 4000,
    trajcl_bounds_margin_deg: float = 0.02,
    trajcl_max_grid_cells: int = 500_000,
    trajcl_seed: int | None = None,
    st2vec_mapper_k: int = 6,
    n_points: int,
    noise_std_deg: float,
    valid_sample_frac: float = 0.15,
    valid_sample_seed: int | None = None,
    valid_n_passes: int = 3,
) -> TrainResult:
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(cfg.device)
    cpu = torch.device("cpu")

    train_pairs, valid_pairs = _load_pairs(sample_ratio, sample_seed)
    train_ds = PairListDataset(train_pairs)
    # Must not use default collate: it would transpose tuples into 7 lists of length B, breaking pair unpacking.
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=lambda batch: batch,
    )

    n_valid = len(valid_pairs)
    if n_valid == 0:
        raise RuntimeError("empty valid set")

    enc = encoder.lower().strip()
    trajgat_helper = None
    st2vec_helper: tuple | None = None
    if enc == "traj2simvec":
        model = Traj2SimVecTwin(rnn_dim=rnn_dim).to(dev)
        tag = "traj2simvec"
    elif enc == "t3s":
        model = T3STwin(
            embed_dim=embed_dim,
            lstm_hidden=t3s_lstm_hidden,
            tf_layers=t3s_tf_layers,
            nhead=t3s_nhead,
            max_len=max(n_points * 2, 64),
        ).to(dev)
        tag = "t3s"
    elif enc == "trajgat":
        _st, _qt, _nid, pre_emb, trajgat_helper = build_trajgat_index(
            train_pairs,
            d_model=trajgat_d_model,
            max_items=trajgat_max_items,
            max_depth=trajgat_max_depth,
            d_lap_pos=trajgat_d_lap,
        )
        model = TrajGATTwin(
            pre_emb,
            d_model=trajgat_d_model,
            d_input=trajgat_d_input,
            num_head=trajgat_nhead,
            num_encoder_layers=trajgat_num_layers,
            d_lap_pos=trajgat_d_lap,
        ).to(dev)
        tag = "trajgat"
    elif enc == "neutraj":
        m_lon, s_lon = lonlat_mean_std_from_pairs(train_pairs)
        model = NeuTrajTwin(
            target_size=rnn_dim,
            lonlat_mean=m_lon,
            lonlat_std=s_lon,
            grid_size=(neutraj_grid_h, neutraj_grid_w),
            recurrent_unit=neutraj_recurrent,
            device=dev,
        ).to(dev)
        tag = "neutraj"
    elif enc == "trajcl":
        min_lon, max_lon, min_lat, max_lat = lonlat_bounds_from_train_pairs(
            train_pairs,
            max_pairs=int(trajcl_bounds_max_pairs),
            margin_deg=float(trajcl_bounds_margin_deg),
        )
        apply_trajcl_config(
            device=dev,
            cell_embedding_dim=int(embed_dim),
            seq_embedding_dim=int(embed_dim),
            moco_proj_dim=int(trajcl_moco_proj_dim),
            moco_nqueue=int(trajcl_moco_nqueue),
            moco_temperature=float(trajcl_moco_temperature),
            cell_size_m=float(trajcl_cell_size_m),
            cellspace_buffer_m=float(trajcl_cellspace_buffer_m),
            trajcl_aug1=str(trajcl_aug1),
            trajcl_aug2=str(trajcl_aug2),
            seed=trajcl_seed if trajcl_seed is not None else int(sample_seed),
        )
        cs, _ = build_cellspace_from_lonlat_bounds(
            min_lon,
            max_lon,
            min_lat,
            max_lat,
            cell_size_m=float(trajcl_cell_size_m),
            buffer_m=float(trajcl_cellspace_buffer_m),
            max_grid_cells=int(trajcl_max_grid_cells),
        )
        model = TrajCLOfficialBaseline(
            cs,
            cell_embedding_dim=int(embed_dim),
            aug1_name=str(trajcl_aug1),
            aug2_name=str(trajcl_aug2),
        ).to(dev)
        tag = "trajcl"
    elif enc == "st2vec":
        if int(embed_dim) % 4 != 0:
            raise ValueError("ST2Vec requires --embed-dim divisible by 4 (Transformer nhead=4).")
        road, mapper = build_st2vec_road_network(
            train_pairs,
            embed_dim=int(embed_dim),
            mapper_k=int(st2vec_mapper_k),
            device=dev,
        )
        model = ST2VecTwin(road, int(embed_dim), str(dev)).to(dev)
        st2vec_helper = (mapper,)
        tag = "st2vec"
    else:
        raise ValueError(
            f"--encoder must be traj2simvec, t3s, trajgat, neutraj, trajcl, or st2vec, got {encoder!r}"
        )

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = -1.0
    best_hr5 = best_hr10 = best_hr20 = best_r5_at_10 = best_r10_at_20 = -1.0
    best_ckpt = None
    stall = 0
    step_noise = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[{tag}] train ep{ep}/{max_epochs}", mininterval=0.5)
        for batch in pbar:
            B = len(batch)
            if B < 2:
                continue
            step_noise += 1
            noise_base = (ep * 1_000_003 + step_noise * 17 + sample_seed * 31) & ((1 << 31) - 1)
            if enc == "trajgat":
                assert trajgat_helper is not None
                g_pre, g_post = batch_pairs_to_graphs(
                    trajgat_helper,
                    batch,
                    n_points=n_points,
                    noise_std_deg=noise_std_deg,
                    noise_base=int(noise_base),
                    device=dev,
                )
                z_pre, z_post = model(g_pre, g_post)
                logit_pos = neg_l2_embedding_logits(z_pre, z_post)
                perm = derangement_indices(B, dev)
                logit_neg = neg_l2_embedding_logits(z_pre, z_post[perm])
                ret = torch.cat([logit_pos, logit_neg], dim=0)
                y = torch.cat([torch.ones(B, device=dev), torch.zeros(B, device=dev)], dim=0)
                loss = pair_loss(ret, y).mean()
            elif enc == "st2vec":
                assert st2vec_helper is not None
                mapper = st2vec_helper[0]
                max_node_idx = int(model.road.x.size(0)) - 1
                pre_ids, pre_te, post_ids, post_te = batch_pairs_to_st2vec_inputs(
                    mapper,
                    batch,
                    n_points=n_points,
                    noise_std_deg=noise_std_deg,
                    noise_base=int(noise_base),
                    embed_dim=int(embed_dim),
                    device=dev,
                    max_node_idx=max_node_idx,
                )
                z_pre, z_post = model(pre_ids, pre_te, post_ids, post_te)
                logit_pos = neg_l2_embedding_logits(z_pre, z_post)
                perm = derangement_indices(B, dev)
                logit_neg = neg_l2_embedding_logits(z_pre, z_post[perm])
                ret = torch.cat([logit_pos, logit_neg], dim=0)
                y = torch.cat([torch.ones(B, device=dev), torch.zeros(B, device=dev)], dim=0)
                loss = pair_loss(ret, y).mean()
            else:
                pre, post = collate_lonlat_noisy(batch, n_points, noise_std_deg, noise_base, cpu)
                pre = pre.to(dev, non_blocking=True)
                post = post.to(dev, non_blocking=True)
                if enc == "trajcl":
                    loss = model.moco_loss_from_post_lonlat(post)
                else:
                    z_pre, z_post = model(pre, post)
                    logit_pos = neg_l2_embedding_logits(z_pre, z_post)
                    perm = derangement_indices(B, dev)
                    logit_neg = neg_l2_embedding_logits(z_pre, z_post[perm])
                    ret = torch.cat([logit_pos, logit_neg], dim=0)
                    y = torch.cat([torch.ones(B, device=dev), torch.zeros(B, device=dev)], dim=0)
                    loss = pair_loss(ret, y).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
        pbar.close()

        # valid: k×k scores = − pairwise L2 distance between embeddings — 3 passes mean
        model.eval()
        k = max(1, min(n_valid, int(round(float(n_valid) * float(valid_sample_frac)))))
        n_passes = max(1, int(valid_n_passes))
        base_seed = int(valid_sample_seed) if valid_sample_seed is not None else (int(sample_seed) + int(ep))

        acc_s = hr5_s = hr10_s = hr20_s = r51_s = r102_s = vloss_s = 0.0
        with torch.no_grad():
            for rep in range(n_passes):
                g_cpu = torch.Generator(device="cpu")
                g_cpu.manual_seed(int(base_seed) + rep * 7919)
                samp = torch.randperm(n_valid, generator=g_cpu)[:k].long()

                noise_base = int(base_seed) * 10007 + rep * 1_000_003 + k * 17
                if enc == "trajgat":
                    assert trajgat_helper is not None
                    g_pre, g_post = graphs_for_valid_indices(
                        trajgat_helper,
                        valid_pairs,
                        samp,
                        n_points=n_points,
                        noise_std_deg=noise_std_deg,
                        noise_base=noise_base,
                        device=dev,
                    )
                    z_pre = model.encode_pre(g_pre)
                    z_post = model.encode_post(g_post)
                elif enc == "st2vec":
                    assert st2vec_helper is not None
                    mapper = st2vec_helper[0]
                    max_node_idx = int(model.road.x.size(0)) - 1
                    pre_ids, pre_te, post_ids, post_te = st2vec_valid_encode_batch(
                        mapper,
                        valid_pairs,
                        samp,
                        n_points=n_points,
                        noise_std_deg=noise_std_deg,
                        noise_base=noise_base,
                        embed_dim=int(embed_dim),
                        device=dev,
                        max_node_idx=max_node_idx,
                    )
                    z_pre = model.encode_pre(pre_ids, pre_te)
                    z_post = model.encode_post(post_ids, post_te)
                else:
                    pres, posts = [], []
                    for j in range(k):
                        idx = int(samp[j].item())
                        pii, poo = pair_pre_post_lonlat12(
                            valid_pairs[idx],
                            n_points=n_points,
                            noise_std_deg=noise_std_deg,
                            noise_seed_pre=noise_base + j * 3,
                            noise_seed_post=noise_base + j * 3 + 1,
                            device=cpu,
                        )
                        pres.append(pii)
                        posts.append(poo)
                    pre_t = torch.stack(pres, dim=0).to(dev)
                    post_t = torch.stack(posts, dim=0).to(dev)

                    z_pre = model.encode_pre(pre_t)
                    z_post = model.encode_post(post_t)
                dist = torch.cdist(z_pre, z_post, p=2)
                scores = -dist

                m = retrieval_metrics(scores)
                vloss_p = float(F.cross_entropy(scores, torch.arange(k, device=dev)).item())

                acc_s += m["acc"]
                hr5_s += m["hr5"]
                hr10_s += m["hr10"]
                hr20_s += m["hr20"]
                r51_s += m["r5_at_10"]
                r102_s += m["r10_at_20"]
                vloss_s += vloss_p

        acc = acc_s / n_passes
        hr5 = hr5_s / n_passes
        hr10 = hr10_s / n_passes
        hr20 = hr20_s / n_passes
        r5_at_10 = r51_s / n_passes
        r10_at_20 = r102_s / n_passes
        valid_loss = vloss_s / n_passes

        if hr5 > best_hr5:
            best_hr5 = hr5
        if hr10 > best_hr10:
            best_hr10 = hr10
        if hr20 > best_hr20:
            best_hr20 = hr20
        if r5_at_10 > best_r5_at_10:
            best_r5_at_10 = r5_at_10
        if r10_at_20 > best_r10_at_20:
            best_r10_at_20 = r10_at_20

        if acc > best_acc:
            best_acc = acc
            stall = 0
            best_ckpt = os.path.join(output_dir, f"best_{tag}_ep{ep}_acc{best_acc:.4f}.pth")
            torch.save(model.state_dict(), best_ckpt)
        else:
            stall += 1
            if stall >= early_stop:
                break

        print(
            f"[{tag}] ep{ep}/{max_epochs} train_loss_mean={sum(losses)/max(1,len(losses)):.6f} "
            f"valid_loss_ce={valid_loss:.6f} (mean over {n_passes} passes @ k={k}/{n_valid} {100.0*k/n_valid:.1f}%) "
            f"valid_acc={acc:.4f} HR5={hr5:.4f} HR10={hr10:.4f} HR20={hr20:.4f} "
            f"R5@10={r5_at_10:.4f} R10@20={r10_at_20:.4f} "
            f"best_acc={best_acc:.4f} stall={stall}/{early_stop}",
            flush=True,
        )

    return TrainResult(
        best_acc=best_acc,
        best_hr5=best_hr5,
        best_hr10=best_hr10,
        best_hr20=best_hr20,
        best_r5_at_10=best_r5_at_10,
        best_r10_at_20=best_r10_at_20,
        best_ckpt=best_ckpt,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--encoder",
        type=str,
        default="traj2simvec",
        choices=("traj2simvec", "t3s", "trajgat", "neutraj", "trajcl", "st2vec"),
        help="Backbone: traj2simvec | t3s | trajgat | neutraj | trajcl | st2vec (repo ST2Vec ST_Encoder).",
    )
    p.add_argument("--train-batch", type=int, default=32)
    p.add_argument("--valid-batch", type=int, default=64, help="Unused (full-k encode); kept for API symmetry.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--early-stop", type=int, default=20)
    p.add_argument("--rnn-dim", type=int, default=128, help="Traj2SimVec LSTM hidden size (subPart dim).")
    p.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Embedding dim: T3S (divisible by --t3s-nhead); ST2Vec (divisible by 4).",
    )
    p.add_argument("--t3s-lstm-hidden", type=int, default=64, help="T3S BiLSTM hidden per direction.")
    p.add_argument("--t3s-tf-layers", type=int, default=2, help="T3S TransformerEncoder layers.")
    p.add_argument("--t3s-nhead", type=int, default=8, help="T3S attention heads.")
    p.add_argument("--trajgat-d-model", type=int, default=128, help="TrajGAT d_model / node2vec dim (must match embedding).")
    p.add_argument("--trajgat-max-items", type=int, default=50, help="Quadtree max items per cell.")
    p.add_argument("--trajgat-max-depth", type=int, default=50, help="Quadtree max depth.")
    p.add_argument("--trajgat-num-layers", type=int, default=3, help="GraphTransformer encoder layers.")
    p.add_argument("--trajgat-nhead", type=int, default=8, help="GraphTransformer attention heads.")
    p.add_argument("--trajgat-d-lap", type=int, default=8, help="Laplacian PE dim (must match loader).")
    p.add_argument("--trajgat-d-input", type=int, default=4, help="Node raw feature dim (official loader uses 4).")
    p.add_argument("--neutraj-grid-h", type=int, default=500, help="NeuTraj grid height (SAM / config gird_size[0]).")
    p.add_argument("--neutraj-grid-w", type=int, default=500, help="NeuTraj grid width (SAM / config gird_size[1]).")
    p.add_argument(
        "--neutraj-recurrent",
        type=str,
        default="LSTM",
        choices=("LSTM", "GRU"),
        help="NeuTraj RNN cell (SimpleRNN omitted: missing SpatialRNNCell in repo). Baseline uses stard LSTM/GRU path.",
    )
    p.add_argument("--trajcl-cell-size", type=float, default=100.0, help="Cell grid size in meters (Mercator), TrajCL vendor default.")
    p.add_argument("--trajcl-cellspace-buffer", type=float, default=500.0, help="Extra meters around lon/lat MBR for CellSpace.")
    p.add_argument("--trajcl-moco-nqueue", type=int, default=2048, help="MoCo queue size (official TrajCL).")
    p.add_argument("--trajcl-moco-proj-dim", type=int, default=128, help="MoCo projector output dim.")
    p.add_argument("--trajcl-moco-temperature", type=float, default=0.05, help="MoCo softmax temperature.")
    p.add_argument("--trajcl-aug1", type=str, default="mask", help="First traj augmentation name (utils.traj.get_aug_fn).")
    p.add_argument("--trajcl-aug2", type=str, default="subset", help="Second traj augmentation name.")
    p.add_argument(
        "--trajcl-bounds-max-pairs",
        type=int,
        default=4000,
        help="Max train pairs to scan when computing lon/lat bounds for CellSpace.",
    )
    p.add_argument("--trajcl-bounds-margin-deg", type=float, default=0.02, help="Margin in degrees added to lon/lat bounds.")
    p.add_argument(
        "--trajcl-max-grid-cells",
        type=int,
        default=500_000,
        help="Cap x_size*y_size for CellSpace; if exceeded, cell size (m) is increased automatically.",
    )
    p.add_argument(
        "--trajcl-seed",
        type=int,
        default=None,
        help="Override TrajCL/vendor RNG seed (default: --sample-seed).",
    )
    p.add_argument(
        "--st2vec-mapper-k",
        type=int,
        default=6,
        help="ST2Vec TrajMapper: decimal places for rounding lon/lat to grid ids (build_graph_from_trajs).",
    )
    p.add_argument("--n-points", type=int, default=12)
    p.add_argument("--noise-std-deg", type=float, default=5e-5)
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--valid-sample-frac", type=float, default=0.15)
    p.add_argument("--valid-n-passes", type=int, default=3)
    p.add_argument("--valid-sample-seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device
    res = run_repr_baseline(
        output_dir=args.output_dir,
        train_batch=int(args.train_batch),
        valid_batch=int(args.valid_batch),
        lr=float(args.lr),
        max_epochs=int(args.max_epochs),
        early_stop=int(args.early_stop),
        sample_ratio=float(args.sample_ratio),
        sample_seed=int(args.sample_seed),
        encoder=str(args.encoder),
        rnn_dim=int(args.rnn_dim),
        embed_dim=int(args.embed_dim),
        t3s_lstm_hidden=int(args.t3s_lstm_hidden),
        t3s_tf_layers=int(args.t3s_tf_layers),
        t3s_nhead=int(args.t3s_nhead),
        trajgat_d_model=int(args.trajgat_d_model),
        trajgat_max_items=int(args.trajgat_max_items),
        trajgat_max_depth=int(args.trajgat_max_depth),
        trajgat_num_layers=int(args.trajgat_num_layers),
        trajgat_nhead=int(args.trajgat_nhead),
        trajgat_d_lap=int(args.trajgat_d_lap),
        trajgat_d_input=int(args.trajgat_d_input),
        neutraj_grid_h=int(args.neutraj_grid_h),
        neutraj_grid_w=int(args.neutraj_grid_w),
        neutraj_recurrent=str(args.neutraj_recurrent),
        trajcl_cell_size_m=float(args.trajcl_cell_size),
        trajcl_cellspace_buffer_m=float(args.trajcl_cellspace_buffer),
        trajcl_moco_nqueue=int(args.trajcl_moco_nqueue),
        trajcl_moco_proj_dim=int(args.trajcl_moco_proj_dim),
        trajcl_moco_temperature=float(args.trajcl_moco_temperature),
        trajcl_aug1=str(args.trajcl_aug1),
        trajcl_aug2=str(args.trajcl_aug2),
        trajcl_bounds_max_pairs=int(args.trajcl_bounds_max_pairs),
        trajcl_bounds_margin_deg=float(args.trajcl_bounds_margin_deg),
        trajcl_max_grid_cells=int(args.trajcl_max_grid_cells),
        trajcl_seed=args.trajcl_seed,
        st2vec_mapper_k=int(args.st2vec_mapper_k),
        n_points=int(args.n_points),
        noise_std_deg=float(args.noise_std_deg),
        valid_sample_frac=float(args.valid_sample_frac),
        valid_sample_seed=args.valid_sample_seed,
        valid_n_passes=int(args.valid_n_passes),
    )
    print(
        f"[{args.encoder}] done "
        f"best_acc={res.best_acc:.4f} best_HR5={res.best_hr5:.4f} best_HR10={res.best_hr10:.4f} "
        f"best_HR20={res.best_hr20:.4f} best_R5@10={res.best_r5_at_10:.4f} best_R10@20={res.best_r10_at_20:.4f} "
        f"ckpt={res.best_ckpt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
