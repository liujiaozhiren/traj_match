"""
Official TrajCL (changyanchuan/TrajCL) integrated for `train_repr_baseline`.

- Vendored under `model_lib/vendor_TrajCL/` (DualSTB + MoCo).
- Cell space in **Web Mercator meters** (same as upstream preprocessing).
- **Trainable** `nn.Embedding` for cells (replaces precomputed node2vec pickle).
- **Training**: MoCo on two augmented views of the **post** trajectory (mask/subset defaults like repo).
- **Inference**: `encode_pre` / `encode_post` use `TrajCL.interpret` (no augmentation).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# --- vendor on path (must precede `import config` / `from model...`) ---
_VENDOR = Path(__file__).resolve().parent / "model_lib" / "vendor_TrajCL"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import config as trajcl_config  # noqa: E402 — vendor root config.py

from model.trajcl import TrajCL  # noqa: E402
from utils.cellspace import CellSpace  # noqa: E402
from utils.tool_funcs import lonlat2meters  # noqa: E402
from utils import traj as trajcl_traj  # noqa: E402


def _lonlat_list_from_tensor(x: torch.Tensor) -> list[list[float]]:
    x = x.detach().float().cpu().numpy()
    return [[float(r[0]), float(r[1])] for r in x]


def _merc_traj_from_lonlat(traj_lonlat: list[list[float]]) -> list[list[float]]:
    return [list(lonlat2meters(p[0], p[1])) for p in traj_lonlat]


def _clip_merc_to_cs(traj: list[list[float]], cs: CellSpace) -> list[list[float]]:
    out = []
    for p in traj:
        x = min(max(p[0], cs.x_min + 1e-6), cs.x_max - 1e-6)
        y = min(max(p[1], cs.y_min + 1e-6), cs.y_max - 1e-6)
        out.append([x, y])
    return out if out else traj


def lonlat_bounds_from_train_pairs(
    pairs: list,
    *,
    max_pairs: int | None = 4000,
    margin_deg: float = 0.02,
) -> tuple[float, float, float, float]:
    """
    Scan pre/post raw trajectories in `pairs` (same layout as `pair_pre_post_lonlat12`)
    and return (min_lon, max_lon, min_lat, max_lat) with margin for CellSpace.

    Envelope uses `raw_traj_lonlat_envelope` (one lon/lat column order per trajectory) so the
    bbox is not inflated by per-point `_choose_lon_lat` mixing axes.
    """
    from cursor_baseline.lonlat_traj import raw_traj_lonlat_envelope
    from cursor_baseline.model_lib.geo_origin import DEFAULT_ORIGIN_LAT, DEFAULT_ORIGIN_LON

    n = len(pairs)
    if n == 0:
        o = 0.5
        return (
            DEFAULT_ORIGIN_LON - o,
            DEFAULT_ORIGIN_LON + o,
            DEFAULT_ORIGIN_LAT - o,
            DEFAULT_ORIGIN_LAT + o,
        )
    if max_pairs is not None and n > max_pairs:
        step = max(1, n // max_pairs)
        indices = list(range(0, n, step))[:max_pairs]
    else:
        indices = list(range(n))

    min_lon = min_lat = math.inf
    max_lon = max_lat = -math.inf
    for i in indices:
        _, A, B, *_ = pairs[i]
        for traj in (A, list(reversed(B))):
            env = raw_traj_lonlat_envelope(traj)
            if env is None:
                continue
            t_min_lon, t_max_lon, t_min_lat, t_max_lat = env
            min_lon = min(min_lon, t_min_lon)
            max_lon = max(max_lon, t_max_lon)
            min_lat = min(min_lat, t_min_lat)
            max_lat = max(max_lat, t_max_lat)

    if not math.isfinite(min_lon):
        o = 0.5
        return (
            DEFAULT_ORIGIN_LON - o,
            DEFAULT_ORIGIN_LON + o,
            DEFAULT_ORIGIN_LAT - o,
            DEFAULT_ORIGIN_LAT + o,
        )

    return (
        min_lon - margin_deg,
        max_lon + margin_deg,
        min_lat - margin_deg,
        max_lat + margin_deg,
    )


def build_cellspace_from_lonlat_bounds(
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    *,
    cell_size_m: float,
    buffer_m: float,
    max_grid_cells: int = 500_000,
) -> tuple[CellSpace, int]:
    """
    Build CellSpace in Web Mercator meters. Uses four lon/lat corners so the Mercator MBR is valid.
    If ``x_size * y_size`` would exceed ``max_grid_cells``, increases the cell edge length (meters)
    until the grid fits (avoids multi‑TB ``nn.Embedding`` tables).
    Returns ``(cellspace, cell_unit_m_int)`` — use the returned unit for vendor ``Config.cell_size``.
    """
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat

    xs: list[float] = []
    ys: list[float] = []
    for lo, la in (
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ):
        x, y = lonlat2meters(lo, la)
        xs.append(x)
        ys.append(y)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_min -= buffer_m
    x_max += buffer_m
    y_min -= buffer_m
    y_max += buffer_m

    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    width = x_max - x_min
    height = y_max - y_min

    cell_u = max(1, int(round(float(cell_size_m))))
    nx = max(1, math.ceil(width / cell_u))
    ny = max(1, math.ceil(height / cell_u))
    n_cells = nx * ny

    if n_cells > max_grid_cells:
        scale = math.sqrt(n_cells / float(max_grid_cells))
        cell_u = max(cell_u, int(math.ceil(cell_u * scale)))
        nx = max(1, math.ceil(width / cell_u))
        ny = max(1, math.ceil(height / cell_u))
        n_cells = nx * ny
        while n_cells > max_grid_cells:
            cell_u += max(1, cell_u // 50)
            nx = max(1, math.ceil(width / cell_u))
            ny = max(1, math.ceil(height / cell_u))
            n_cells = nx * ny

    if cell_u != int(round(float(cell_size_m))):
        print(
            f"[trajcl] CellSpace grid {nx}x{ny}={n_cells} cells; raised cell size (m) "
            f"{int(round(float(cell_size_m)))} -> {cell_u} to stay under max_grid_cells={max_grid_cells}.",
            flush=True,
        )

    cs = CellSpace(cell_u, cell_u, x_min, y_min, x_max, y_max)
    trajcl_config.Config.cell_size = float(cell_u)
    trajcl_config.Config.trajcl_local_mask_sidelen = float(cell_u) * 11.0
    return cs, cell_u


def apply_trajcl_config(
    *,
    device: torch.device,
    cell_embedding_dim: int,
    seq_embedding_dim: int,
    moco_proj_dim: int,
    moco_nqueue: int,
    moco_temperature: float,
    cell_size_m: float,
    cellspace_buffer_m: float,
    trajcl_aug1: str,
    trajcl_aug2: str,
    seed: int | None = None,
) -> None:
    C = trajcl_config.Config
    C.device = device
    if seed is not None:
        C.seed = int(seed)
    C.cell_embedding_dim = cell_embedding_dim
    C.seq_embedding_dim = seq_embedding_dim
    C.moco_proj_dim = moco_proj_dim
    C.moco_nqueue = moco_nqueue
    C.moco_temperature = moco_temperature
    C.cell_size = float(cell_size_m)
    C.cellspace_buffer = float(cellspace_buffer_m)
    C.trajcl_aug1 = trajcl_aug1
    C.trajcl_aug2 = trajcl_aug2
    C.trajcl_local_mask_sidelen = float(cell_size_m) * 11.0
    trajcl_config.set_seed(trajcl_config.Config.seed)


class TrajCLOfficialBaseline(nn.Module):
    """
    Wraps official `TrajCL` + trainable cell embeddings.
    """

    def __init__(
        self,
        cellspace: CellSpace,
        *,
        cell_embedding_dim: int,
        aug1_name: str = "mask",
        aug2_name: str = "subset",
    ):
        super().__init__()
        self.cellspace = cellspace
        n_cells = cellspace.x_size * cellspace.y_size
        self.cell_emb = nn.Embedding(n_cells, cell_embedding_dim)
        self.aug1 = trajcl_traj.get_aug_fn(aug1_name)
        self.aug2 = trajcl_traj.get_aug_fn(aug2_name)
        if self.aug1 is None or self.aug2 is None:
            raise ValueError(f"unknown aug names {aug1_name} {aug2_name}")
        self.core = TrajCL()

    def num_cells(self) -> int:
        return int(self.cell_emb.num_embeddings)

    def _cells_to_emb_seq(self, cells: tuple, device: torch.device) -> torch.Tensor:
        idx = torch.tensor(list(cells), dtype=torch.long, device=device)
        return self.cell_emb(idx)

    def _pack_two_views(
        self,
        trajs_merc: list[list[list[float]]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trajs1 = [self.aug1(list(t)) for t in trajs_merc]
        trajs2 = [self.aug2(list(t)) for t in trajs_merc]
        emb1_cell: list[torch.Tensor] = []
        emb1_sp: list[torch.Tensor] = []
        emb2_cell: list[torch.Tensor] = []
        emb2_sp: list[torch.Tensor] = []
        cs = self.cellspace
        for t1, t2 in zip(trajs1, trajs2):
            t1 = _clip_merc_to_cs(t1, cs)
            t2 = _clip_merc_to_cs(t2, cs)
            c1, p1 = trajcl_traj.merc2cell2(t1, cs)
            c2, p2 = trajcl_traj.merc2cell2(t2, cs)
            if len(c1) == 0 or len(c2) == 0:
                c1 = (0,)
                c2 = (0,)
                p1 = [t1[0]] if t1 else [[cs.x_min, cs.y_min]]
                p2 = [t2[0]] if t2 else [[cs.x_min, cs.y_min]]
            # guard length-1 sequences (same as _pack_single_view): the dual-attention
            # key_padding_mask needs >=2 tokens, else its shape mismatches src_len.
            if len(c1) == 1:
                c1 = (c1[0], c1[0])
                p1 = list(p1) + [p1[0]]
            if len(c2) == 1:
                c2 = (c2[0], c2[0])
                p2 = list(p2) + [p2[0]]
            emb1_cell.append(self._cells_to_emb_seq(c1, device))
            emb2_cell.append(self._cells_to_emb_seq(c2, device))
            sp1 = torch.tensor(trajcl_traj.generate_spatial_features(p1, cs), dtype=torch.float32, device=device)
            sp2 = torch.tensor(trajcl_traj.generate_spatial_features(p2, cs), dtype=torch.float32, device=device)
            emb1_sp.append(sp1)
            emb2_sp.append(sp2)
        trajs1_emb = pad_sequence(emb1_cell, batch_first=False)
        trajs2_emb = pad_sequence(emb2_cell, batch_first=False)
        trajs1_emb_p = pad_sequence(emb1_sp, batch_first=False)
        trajs2_emb_p = pad_sequence(emb2_sp, batch_first=False)
        trajs1_len = torch.tensor([len(e) for e in emb1_cell], dtype=torch.long, device=device)
        trajs2_len = torch.tensor([len(e) for e in emb2_cell], dtype=torch.long, device=device)
        return trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len

    def _pack_single_view(
        self,
        trajs_merc: list[list[list[float]]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_cell: list[torch.Tensor] = []
        emb_sp: list[torch.Tensor] = []
        cs = self.cellspace
        for t in trajs_merc:
            t = _clip_merc_to_cs(t, cs)
            c, p = trajcl_traj.merc2cell2(t, cs)
            if len(c) == 0:
                c = (0,)
                p = [t[0]] if t else [[cs.x_min, cs.y_min]]
            if len(c) == 1:
                c = (c[0], c[0])
                p = list(p) + [p[0]]
            emb_cell.append(self._cells_to_emb_seq(c, device))
            emb_sp.append(torch.tensor(trajcl_traj.generate_spatial_features(p, cs), dtype=torch.float32, device=device))
        trajs_emb = pad_sequence(emb_cell, batch_first=False)
        trajs_emb_p = pad_sequence(emb_sp, batch_first=False)
        trajs_len = torch.tensor([len(e) for e in emb_cell], dtype=torch.long, device=device)
        return trajs_emb, trajs_emb_p, trajs_len

    def moco_loss_from_post_lonlat(self, post: torch.Tensor) -> torch.Tensor:
        """post: (B, L, 2) lon/lat degrees — MoCo on two augmented Mercator views."""
        device = post.device
        B = post.size(0)
        trajs_merc = [_merc_traj_from_lonlat(_lonlat_list_from_tensor(post[i])) for i in range(B)]
        t1, t1p, l1, t2, t2p, l2 = self._pack_two_views(trajs_merc, device)
        logits, targets = self.core(t1, t1p, l1, t2, t2p, l2)
        return self.core.loss(logits, targets)

    def _encode_lonlat_batch(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        B = x.size(0)
        trajs_merc = [_merc_traj_from_lonlat(_lonlat_list_from_tensor(x[i])) for i in range(B)]
        te, tep, tl = self._pack_single_view(trajs_merc, device)
        return self.core.interpret(te, tep, tl)

    def encode_pre(self, pre: torch.Tensor) -> torch.Tensor:
        return self._encode_lonlat_batch(pre)

    def encode_post(self, post: torch.Tensor) -> torch.Tensor:
        return self._encode_lonlat_batch(post)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_pre(pre), self.encode_post(post)
