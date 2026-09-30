"""All learning-based match models (self-contained, embed_dim=128)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import (
    DIFFUSION_CH,
    DIFFUSION_STEPS,
    EMBED_DIM,
    HYDRA_STEP_LEN,
    HYDRA_TAIL,
    INFO_LEN,
    K_SELECT_RL,
    K_SELECT_VANILLA,
    MLP_HIDDEN,
    N_HEADS,
    RNN_HIDDEN,
)
from .common import (
    ChainGnnEncoder,
    GruLastStepEncoder,
    GruTrajEncoder,
    NeuTrajEncoder,
    PairMlpScorer,
    T3SEncoder,
    TwinEncoder,
)


# ----- representation twins -----


class Traj2SimVecTwin(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.enc_pre = GruLastStepEncoder(embed_dim)
        self.enc_post = GruLastStepEncoder(embed_dim)

    def encode_pre(self, x):
        return self.enc_pre(x)

    def encode_post(self, x):
        return self.enc_post(x)


class T3STwin(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.enc_pre = T3SEncoder(embed_dim)
        self.enc_post = T3SEncoder(embed_dim)

    def encode_pre(self, x):
        return self.enc_pre(x)

    def encode_post(self, x):
        return self.enc_post(x)


class NeuTrajTwin(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.enc_pre = NeuTrajEncoder(embed_dim)
        self.enc_post = NeuTrajEncoder(embed_dim)

    def encode_pre(self, x):
        return self.enc_pre(x)

    def encode_post(self, x):
        return self.enc_post(x)


class TrajGATTwin(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, aux_nodes: int = 256):
        super().__init__()
        self.enc_pre = ChainGnnEncoder(embed_dim)
        self.enc_post = ChainGnnEncoder(embed_dim)
        self.pre_embedding = nn.Parameter(torch.randn(aux_nodes, embed_dim) * 0.02)

    def encode_pre(self, x):
        return self.enc_pre(x)

    def encode_post(self, x):
        return self.enc_post(x)


class TrajCLTwin(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, n_cells: int = 4096):
        super().__init__()
        self.cell_emb = nn.Embedding(n_cells, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.enc_pre = None  # shared weights via methods
        self.enc_post = None
        self.n_cells = n_cells
        self._build()

    def _build(self):
        self.proj = nn.Linear(2, EMBED_DIM)

    def _traj_to_cell_ids(self, x: torch.Tensor) -> torch.Tensor:
        lon = x[..., 0]
        lat = x[..., 1]
        cid = (
            ((lon - lon.min()) / (lon.max() - lon.min() + 1e-6) * 63).long() * 64
            + ((lat - lat.min()) / (lat.max() - lat.min() + 1e-6) * 63).long()
        )
        return cid.clamp(0, self.n_cells - 1)

    def encode_pre(self, x):
        return self._encode(x)

    def encode_post(self, x):
        return self._encode(x)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        ids = self._traj_to_cell_ids(x)
        emb = self.cell_emb(ids)
        out, _ = self.gru(emb)
        return out[:, -1, :]


class ST2VecTwin(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, n_nodes: int = 512):
        super().__init__()
        self.road_x = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.02)
        self.enc = ChainGnnEncoder(embed_dim)

    def encode_pre(self, x):
        return self.enc(x)

    def encode_post(self, x):
        return self.enc(x)


# ----- DAN / TAT -----


class DANMotTraj(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = GruTrajEncoder(embed_dim)
        self.pair = PairMlpScorer(embed_dim, MLP_HIDDEN)

    def encode_pre(self, x):
        return self.backbone(x)

    def encode_post(self, x):
        return self.backbone(x)

    def score_gallery(self, u_q: torch.Tensor, u_gallery: torch.Tensor) -> torch.Tensor:
        return self.pair.score_all(u_q, u_gallery)


class TATTraj(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, use_graph_update: bool = False):
        super().__init__()
        self.encoder_pre = GruTrajEncoder(embed_dim)
        self.encoder_post = GruTrajEncoder(embed_dim)
        self.use_graph_update = use_graph_update
        self.graph = (
            nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True) if use_graph_update else None
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pair = PairMlpScorer(embed_dim, MLP_HIDDEN)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_pre(x)

    def encode_post(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_post(x)

    def _pair_embed(self, h_pre: torch.Tensor, h_post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.graph is None:
            return h_pre, h_post
        nodes = torch.stack([h_pre, h_post], dim=1)
        out, _ = self.graph(nodes, nodes, nodes, need_weights=False)
        nodes = self.norm(nodes + out)
        return nodes[:, 0], nodes[:, 1]

    def score_gallery_batched(self, h_q: torch.Tensor, h_gallery: torch.Tensor) -> torch.Tensor:
        n = h_gallery.shape[0]
        if not self.use_graph_update:
            return self.pair.score_all(h_q, h_gallery)
        h_pre = h_q.unsqueeze(0).expand(n, -1)
        hp, hg = self._pair_embed(h_pre, h_gallery)
        pair = torch.cat([hp, hg], dim=-1)
        return self.pair.mlp(self.pair.ln(pair)).squeeze(-1)


# ----- AttnMove -----


class RegionVocab(nn.Module):
    def __init__(self, n_regions: int = 256, hidden: int = EMBED_DIM):
        super().__init__()
        self.n_regions = n_regions
        self.centroids = nn.Parameter(torch.rand(n_regions, 2))
        self.embed = nn.Embedding(n_regions, hidden)

    def lonlat_to_ids(self, xy: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(xy, self.centroids)
        return d.argmin(dim=-1)

    def ids_to_lonlat(self, ids: torch.Tensor) -> torch.Tensor:
        return self.centroids[ids.clamp(0, self.n_regions - 1)]


class AttnMoveCompletion(nn.Module):
    def __init__(self, vocab: RegionVocab, hidden: int = EMBED_DIM):
        super().__init__()
        self.vocab = vocab
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.out = nn.Linear(hidden, 2)

    def forward(self, a_xy: torch.Tensor) -> torch.Tensor:
        ids = self.vocab.lonlat_to_ids(a_xy)
        mem = self.vocab.embed(ids)
        q = mem
        out, _ = self.attn(q, mem, mem, need_weights=False)
        h = self.ff(out + q)
        return self.out(h)


def completion_gallery_scores(pred: torch.Tensor, gallery_post: torch.Tensor) -> torch.Tensor:
    """pred (T,2), gallery (N,T,2) -> (N,) higher=better."""
    diff = pred.unsqueeze(0).unsqueeze(2) - gallery_post.unsqueeze(1)
    d = torch.linalg.vector_norm(diff, dim=-1).mean(dim=-1)
    return -d.squeeze(1)


# ----- mod4: mini diffusion + scheme4 match -----


class MiniTrajUNet1d(nn.Module):
    def __init__(self, ch: int = DIFFUSION_CH):
        super().__init__()
        self.ch = ch
        self.temb = nn.Sequential(nn.Linear(ch, ch * 4), nn.SiLU(), nn.Linear(ch * 4, ch * 4))
        self.in_x = nn.Conv1d(2, ch, 3, padding=1)
        self.in_info = nn.Conv1d(2, ch, 3, padding=1)
        self.mid = nn.Conv1d(ch * 2, ch, 3, padding=1)
        self.out = nn.Conv1d(ch, 2, 3, padding=1)

    def forward(self, xt: torch.Tensor, info: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.temb(self._temb(t))
        hx = self.in_x(xt)
        hi = self.in_info(info)
        if hi.shape[-1] != hx.shape[-1]:
            hi = F.interpolate(hi, size=hx.shape[-1], mode="linear", align_corners=False)
        b, c, l = hx.shape
        temb4 = temb.unsqueeze(-1).expand(b, temb.shape[1], l)[:, : c * 2]
        h = torch.cat([hx, hi], dim=1)
        if temb4.shape[1] == h.shape[1]:
            h = h + temb4
        h = self.mid(h)
        return self.out(h)

    @staticmethod
    def _temb(t: torch.Tensor) -> torch.Tensor:
        half = DIFFUSION_CH // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if DIFFUSION_CH % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class FixLenDiff84(nn.Module):
    """DDPM infer for hydra tails (self-contained)."""

    def __init__(self, n_steps: int = DIFFUSION_STEPS, device: torch.device | None = None):
        super().__init__()
        self.n_steps = int(n_steps)
        self.unet = MiniTrajUNet1d()
        beta = torch.linspace(1e-4, 0.05, self.n_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def infer_from_noise(self, info: torch.Tensor, num: int) -> torch.Tensor:
        """info (B,2,L_info) -> (B,num,2,L_tail)."""
        b, _, _ = info.shape
        dev = info.device
        l_tail = HYDRA_STEP_LEN
        last = self.n_steps - 1
        info_rep = info.repeat_interleave(int(num), dim=0)
        xt = torch.randn(b * int(num), 2, l_tail, device=dev)
        use_bf16 = dev.type == "cuda"
        for step in range(last, -1, -1):
            t = torch.full((xt.shape[0],), step, device=dev, dtype=torch.long)
            if use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    eps = self.unet(xt, info_rep, t)
            else:
                eps = self.unet(xt, info_rep, t)
            ab = self.alpha_bar[t].view(-1, 1, 1)
            a = self.alpha[t].view(-1, 1, 1)
            x0 = (xt - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-8)
            if step > 0:
                xt = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn_like(xt)
            else:
                xt = x0
        return xt.reshape(b, int(num), 2, l_tail)


class Scheme4PairTransformer(nn.Module):
    def __init__(self, dim: int = EMBED_DIM, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.point = nn.Linear(2, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Sequential(nn.Linear(dim, MLP_HIDDEN), nn.ReLU(), nn.Linear(MLP_HIDDEN, 1))

    def _flat(self, head: torch.Tensor, hydra: torch.Tensor) -> torch.Tensor:
        b = head.size(0)
        tail = hydra.reshape(b, -1, 2)
        return torch.cat([head, tail], dim=1)

    def forward(self, info_pre, info_post, hydra_pre, hydra_post) -> torch.Tensor:
        a = self._flat(info_pre, hydra_pre)
        b = self._flat(info_post, hydra_post)
        x = torch.cat([a, b], dim=1)
        x = self.point(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        h = self.enc(x)[:, 0, :]
        return self.out(h).squeeze(-1)


class TailSelector(nn.Module):
    """RL selector: score hydra tails, pick top k_select."""

    def __init__(self, rep_dim: int = 32, k_select: int = K_SELECT_RL):
        super().__init__()
        self.k_select = int(k_select)
        in_dim = INFO_LEN * 2 * 2 + HYDRA_STEP_LEN * 2 * 2
        self.enc = nn.Linear(in_dim, rep_dim)
        self.scorer = nn.Linear(rep_dim, 1)

    def choose(
        self,
        info_pre: torch.Tensor,
        info_post: torch.Tensor,
        hydra_pre: torch.Tensor,
        hydra_post: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """hydra (B,N_tails,k,2) -> selected (B,k_sel,k,2) each side; joint index."""
        b, n, k, _ = hydra_pre.shape
        ks = min(self.k_select, n)
        flat_pre = hydra_pre.reshape(b, n, -1)
        flat_post = hydra_post.reshape(b, n, -1)
        ctx = torch.cat(
            [
                info_pre.unsqueeze(1).expand(-1, n, -1, -1).reshape(b, n, -1),
                info_post.unsqueeze(1).expand(-1, n, -1, -1).reshape(b, n, -1),
                flat_pre,
                flat_post,
            ],
            dim=-1,
        )
        s = self.scorer(F.relu(self.enc(ctx))).squeeze(-1)
        idx = torch.topk(s, k=ks, dim=1).indices
        idx_expand = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k, 2)
        h_pre = torch.gather(hydra_pre, 1, idx_expand)
        h_post = torch.gather(hydra_post, 1, idx_expand)
        return h_pre, h_post


class Mod4Pipeline(nn.Module):
    def __init__(self, use_selector: bool = False):
        super().__init__()
        self.pre_diff = FixLenDiff84()
        self.post_diff = FixLenDiff84()
        self.match = Scheme4PairTransformer()
        self.selector = TailSelector() if use_selector else None
        self.use_selector = use_selector

    def infer_query_hydra(self, head_pre: torch.Tensor, *, hydra_tail: int) -> torch.Tensor:
        """Online only: single query pre-side diffusion."""
        info = head_pre.unsqueeze(0)
        out = self.pre_diff.infer_from_noise(info, num=hydra_tail)
        return out[0].permute(0, 2, 1).contiguous()

    def infer_gallery_hydra(self, head_post: torch.Tensor, *, hydra_tail: int) -> torch.Tensor:
        """Offline index build (single item). Prefer infer_gallery_hydra_batch."""
        return self.infer_gallery_hydra_batch(head_post.unsqueeze(0), hydra_tail=hydra_tail)[0]

    def infer_gallery_hydra_batch(self, heads_post: torch.Tensor, *, hydra_tail: int) -> torch.Tensor:
        """Offline: batch post diffusion -> (B, hydra_tail, k, 2)."""
        info = heads_post  # (B, 2, L)
        out = self.post_diff.infer_from_noise(info, num=hydra_tail)
        return out.permute(0, 1, 3, 2).contiguous()

    def match_gallery_batch(
        self,
        info_q: torch.Tensor,
        hydra_q: torch.Tensor,
        info_g: torch.Tensor,
        hydra_g: torch.Tensor,
    ) -> torch.Tensor:
        """Single query vs N gallery, already batched (N,...)."""
        n = info_g.shape[0]
        iq = info_q.unsqueeze(0).expand(n, -1, -1)
        hq = hydra_q.unsqueeze(0).expand(n, *hydra_q.shape)
        if self.selector is not None:
            hq, hg = self.selector.choose(iq, info_g, hq, hydra_g)
        else:
            ks = min(K_SELECT_VANILLA, hq.shape[1])
            hq, hg = hq[:, :ks], hydra_g[:, :ks]
        return self.match(iq, info_g, hq, hg)


def build_model(name: str) -> nn.Module:
    name = name.lower()
    table = {
        "traj2simvec": Traj2SimVecTwin,
        "t3s": T3STwin,
        "neutraj": NeuTrajTwin,
        "trajgat": TrajGATTwin,
        "trajcl": TrajCLTwin,
        "st2vec": ST2VecTwin,
        "dan": DANMotTraj,
        "tat": lambda: TATTraj(use_graph_update=False),
        "tat_graph": lambda: TATTraj(use_graph_update=True),
        "attnmove": lambda: AttnMoveCompletion(RegionVocab()),
        "mod4": lambda: Mod4Pipeline(use_selector=False),
        "mod4_rl": lambda: Mod4Pipeline(use_selector=True),
    }
    if name not in table:
        raise KeyError(f"unknown model {name}")
    factory = table[name]
    return factory() if callable(factory) and not isinstance(factory, type) else factory()
