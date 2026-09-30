"""Full ``cursor_baseline`` pipelines for match-efficiency benchmarks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

sys.modules.setdefault("cfg", importlib.import_module("cursor_baseline.model_lib.cfg"))

from cursor_baseline.attnmove_model import AttnMoveCompletion  # noqa: E402
from cursor_baseline.attnmove_region import RegionVocab  # noqa: E402
from cursor_baseline.dan_mot_traj.model import DANMotTrajSimplified  # noqa: E402
from cursor_baseline.lonlat_traj import pair_pre_post_lonlat12  # noqa: E402
from cursor_baseline.neutraj_twin import NeuTrajTwin, lonlat_bounds_from_pairs, lonlat_mean_std_from_pairs  # noqa: E402
from cursor_baseline.st2vec_build import batch_pairs_to_st2vec_inputs, build_st2vec_road_network  # noqa: E402
from cursor_baseline.st2vec_vendor_load import get_st_encoder_class  # noqa: E402
from cursor_baseline.t3s_twin import T3STwin  # noqa: E402
from cursor_baseline.tat_traj.model import TATTrajTwoSegmentBaseline  # noqa: E402
from cursor_baseline.traj2simvec_twin import Traj2SimVecTwin  # noqa: E402
from cursor_baseline.trajcl_official import (  # noqa: E402
    TrajCLOfficialBaseline,
    apply_trajcl_config,
    build_cellspace_from_lonlat_bounds,
    lonlat_bounds_from_train_pairs,
)
from cursor_baseline.trajgat_build import batch_pairs_to_graphs, build_trajgat_index  # noqa: E402
from cursor_baseline.trajgat_twin import TrajGATTwin  # noqa: E402

from .constants import (  # noqa: E402
    ATTNMOVE_HIDDEN,
    ATTNMOVE_LAYERS,
    DAN_RNN_LAYERS,
    DAN_USE_UNMATCHED_DIM,
    EMBED_DIM,
    ENCODE_BATCH,
    MLP_HIDDEN,
    NEUTRAJ_GRID,
    NEUTRAJ_RECURRENT,
    NEUTRAJ_SAM_LAYERS,
    NEUTRAJ_STARD_LSTM,
    NOISE_STD_DEG,
    N_POINTS,
    RNN_HIDDEN,
    ST2VEC_NUM_LAYERS,
    T3S_TF_LAYERS,
    TAT_RNN_LAYERS,
    TRAJ2SIMVEC_LSTM_LAYERS,
    TRAJ2SIMVEC_SUBPART_BLOCKS,
    TRAJCL_TRANS_LAYERS,
    TRAJGAT_NUM_LAYERS,
)
from .models.common import batched_cdist_scores  # noqa: E402

ST_Encoder = get_st_encoder_class()

TENSOR_REPR = frozenset({"traj2simvec", "t3s", "neutraj", "trajgat", "trajcl", "st2vec"})
PAIR_MLP = frozenset({"dan", "tat", "tat_graph"})
FULL_BASELINE = TENSOR_REPR | PAIR_MLP | frozenset({"attnmove"})


class ST2VecTwinDeep(nn.Module):
    """Official ST_Encoder with configurable BiLSTM depth."""

    def __init__(self, road_network, embed_dim: int, device_str: str, *, num_layers: int = ST2VEC_NUM_LAYERS):
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("ST2Vec embed_dim must be divisible by 4")
        self.road = road_network
        self.embed_dim = int(embed_dim)
        kw = dict(
            feature_size=embed_dim,
            date2vec_size=embed_dim,
            embedding_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=int(num_layers),
            dropout_rate=0,
            device=device_str,
        )
        self.pre = ST_Encoder(**kw)
        self.post = ST_Encoder(**kw)

    def _net(self):
        d = next(self.parameters()).device
        return self.road.to(d)

    def encode_pre(self, traj_seqs: list, time_seqs: list):
        return self.pre(self._net(), traj_seqs, time_seqs)

    def encode_post(self, traj_seqs: list, time_seqs: list):
        return self.post(self._net(), traj_seqs, time_seqs)


def _sync_trajcl_device(device: torch.device) -> None:
    _vendor = _REPO / "cursor_baseline" / "model_lib" / "vendor_TrajCL"
    if str(_vendor) not in sys.path:
        sys.path.insert(0, str(_vendor))
    import config as trajcl_config  # noqa: E402

    trajcl_config.Config.device = device


def _pair_items(pairs: list[tuple], indices: list[int]) -> list[tuple]:
    return [pairs[i] for i in indices]


def _stack_lonlat_posts(pairs: list[tuple], indices: list[int], device: torch.device) -> torch.Tensor:
    posts = []
    cpu = torch.device("cpu")
    for j, gi in enumerate(indices):
        _, post = pair_pre_post_lonlat12(
            pairs[gi],
            n_points=N_POINTS,
            noise_std_deg=NOISE_STD_DEG,
            noise_seed_pre=gi,
            noise_seed_post=gi,
            device=cpu,
        )
        posts.append(post)
    return torch.stack(posts, dim=0).to(device)


def _chunked(n: int, batch: int):
    for i0 in range(0, n, batch):
        yield i0, min(i0 + batch, n)


def build_artifacts(method: str, pairs: list[tuple], device: torch.device) -> dict[str, Any]:
    method = method.lower()
    art: dict[str, Any] = {"method": method}
    if method == "neutraj":
        m, s = lonlat_mean_std_from_pairs(pairs)
        art["lonlat_mean"] = m
        art["lonlat_std"] = s
        art["lonlat_bounds"] = lonlat_bounds_from_pairs(pairs)
    elif method == "trajgat":
        _st, _qt, _nid, pre_emb, helper = build_trajgat_index(pairs, d_model=EMBED_DIM)
        art["trajgat_helper"] = helper
        art["pre_embedding"] = pre_emb
    elif method == "trajcl":
        min_lon, max_lon, min_lat, max_lat = lonlat_bounds_from_train_pairs(pairs)
        _vendor = _REPO / "cursor_baseline" / "model_lib" / "vendor_TrajCL"
        if str(_vendor) not in sys.path:
            sys.path.insert(0, str(_vendor))
        import config as trajcl_config  # noqa: E402

        apply_trajcl_config(
            device=device,
            cell_embedding_dim=EMBED_DIM,
            seq_embedding_dim=EMBED_DIM,
            moco_proj_dim=EMBED_DIM,
            moco_nqueue=2048,
            moco_temperature=0.05,
            cell_size_m=100.0,
            cellspace_buffer_m=500.0,
            trajcl_aug1="mask",
            trajcl_aug2="subset",
            seed=42,
        )
        trajcl_config.Config.trans_attention_layer = int(TRAJCL_TRANS_LAYERS)
        cs, _ = build_cellspace_from_lonlat_bounds(
            min_lon, max_lon, min_lat, max_lat, cell_size_m=100.0, buffer_m=500.0
        )
        art["cellspace"] = cs
    elif method == "st2vec":
        road, mapper = build_st2vec_road_network(pairs, embed_dim=EMBED_DIM, mapper_k=6, device=device)
        art["road"] = road
        art["mapper"] = mapper
        art["max_node_idx"] = int(road.x.size(0) - 1)
    elif method == "attnmove":
        art["region_vocab"] = _fit_attnmove_vocab(pairs)
    return art


def _fit_attnmove_vocab(pairs: list) -> RegionVocab:
    """Grid regions from subsampled lon/lat (same length pre/post)."""
    coords: list[tuple[float, float]] = []
    cpu = torch.device("cpu")
    for i, item in enumerate(pairs):
        pre, post = pair_pre_post_lonlat12(
            item,
            n_points=N_POINTS,
            noise_std_deg=NOISE_STD_DEG,
            noise_seed_pre=i,
            noise_seed_post=i,
            device=cpu,
        )
        for t in range(pre.shape[0]):
            coords.append((float(pre[t, 0].item()), float(pre[t, 1].item())))
            coords.append((float(post[t, 0].item()), float(post[t, 1].item())))
    if not coords:
        raise RuntimeError("RegionVocab: no coordinates from pairs")
    stub = [
        (
            "stub",
            [[0, c[0], c[1]] for c in coords],
            [[0, c[0], c[1]] for c in coords],
            [],
            [],
            None,
            None,
        )
    ]
    return RegionVocab.fit_from_pairs(stub, cpu=cpu)


def build_model(method: str, artifacts: dict[str, Any], device: torch.device) -> nn.Module:
    method = method.lower()
    if method == "traj2simvec":
        return Traj2SimVecTwin(
            rnn_dim=RNN_HIDDEN,
            num_lstm_layers=TRAJ2SIMVEC_LSTM_LAYERS,
            num_subpart_blocks=TRAJ2SIMVEC_SUBPART_BLOCKS,
        ).to(device)
    if method == "t3s":
        return T3STwin(
            embed_dim=EMBED_DIM,
            lstm_hidden=RNN_HIDDEN,
            tf_layers=T3S_TF_LAYERS,
            nhead=8,
            max_len=max(N_POINTS * 2, 64),
        ).to(device)
    if method == "neutraj":
        return NeuTrajTwin(
            target_size=EMBED_DIM,
            lonlat_mean=artifacts["lonlat_mean"],
            lonlat_std=artifacts["lonlat_std"],
            lonlat_bounds=artifacts.get("lonlat_bounds"),
            grid_size=NEUTRAJ_GRID,
            recurrent_unit=NEUTRAJ_RECURRENT,
            stard_lstm=NEUTRAJ_STARD_LSTM,
            num_layers=NEUTRAJ_SAM_LAYERS,
            device=device,
        ).to(device)
    if method == "trajgat":
        return TrajGATTwin(
            artifacts["pre_embedding"],
            d_model=EMBED_DIM,
            num_encoder_layers=TRAJGAT_NUM_LAYERS,
        ).to(device)
    if method == "trajcl":
        return TrajCLOfficialBaseline(artifacts["cellspace"], cell_embedding_dim=EMBED_DIM).to(device)
    if method == "st2vec":
        return ST2VecTwinDeep(
            artifacts["road"],
            EMBED_DIM,
            str(device),
            num_layers=ST2VEC_NUM_LAYERS,
        ).to(device)
    if method == "dan":
        return DANMotTrajSimplified(
            rnn_hidden=RNN_HIDDEN,
            embed_dim=EMBED_DIM,
            rnn_layers=DAN_RNN_LAYERS,
            mlp_hidden=MLP_HIDDEN,
            use_unmatched_dim=DAN_USE_UNMATCHED_DIM,
        ).to(device)
    if method in ("tat", "tat_graph"):
        return TATTrajTwoSegmentBaseline(
            rnn_hidden=RNN_HIDDEN,
            embed_dim=EMBED_DIM,
            rnn_layers=TAT_RNN_LAYERS,
            mlp_hidden=MLP_HIDDEN,
            use_graph_update=True,
        ).to(device)
    if method == "attnmove":
        return AttnMoveCompletion(
            artifacts["region_vocab"],
            hidden=ATTNMOVE_HIDDEN,
            n_layers=ATTNMOVE_LAYERS,
            n_heads=4,
        ).to(device)
    raise ValueError(f"unsupported full baseline method: {method}")


@torch.no_grad()
def encode_posts_batched(
    method: str,
    model: nn.Module,
    artifacts: dict[str, Any],
    pairs: list[tuple],
    gallery_indices: list[int],
    *,
    device: torch.device,
    encode_batch: int = ENCODE_BATCH,
) -> torch.Tensor | None:
    """Return (N,D) embeddings for repr / pair methods; None for attnmove."""
    method = method.lower()
    n = len(gallery_indices)
    if method in TENSOR_REPR:
        if method in ("traj2simvec", "t3s", "neutraj", "trajcl"):
            if method == "trajcl":
                _sync_trajcl_device(device)
            posts = _stack_lonlat_posts(pairs, gallery_indices, device)
            outs = []
            for i0, i1 in _chunked(n, encode_batch):
                outs.append(model.encode_post(posts[i0:i1]))
            return torch.cat(outs, dim=0)
        if method == "trajgat":
            helper = artifacts["trajgat_helper"]
            outs = []
            for i0, i1 in _chunked(n, encode_batch):
                items = _pair_items(pairs, gallery_indices[i0:i1])
                _, g_post = batch_pairs_to_graphs(
                    helper,
                    items,
                    n_points=N_POINTS,
                    noise_std_deg=NOISE_STD_DEG,
                    noise_base=i0,
                    device=device,
                )
                outs.append(model.encode_post(g_post))
            return torch.cat(outs, dim=0)
        if method == "st2vec":
            mapper = artifacts["mapper"]
            max_node_idx = artifacts["max_node_idx"]
            outs = []
            for i0, i1 in _chunked(n, encode_batch):
                items = _pair_items(pairs, gallery_indices[i0:i1])
                _pre_ids, _pre_te, post_ids, post_te = batch_pairs_to_st2vec_inputs(
                    mapper,
                    items,
                    n_points=N_POINTS,
                    noise_std_deg=NOISE_STD_DEG,
                    noise_base=i0,
                    embed_dim=EMBED_DIM,
                    device=device,
                    max_node_idx=max_node_idx,
                )
                outs.append(model.encode_post(post_ids, post_te))
            return torch.cat(outs, dim=0)
    if method in PAIR_MLP:
        posts = _stack_lonlat_posts(pairs, gallery_indices, device)
        outs = []
        enc = model.encoder_post if method in ("tat", "tat_graph") else model.backbone
        for i0, i1 in _chunked(n, encode_batch):
            outs.append(enc(posts[i0:i1]))
        return torch.cat(outs, dim=0)
    return None


@torch.no_grad()
def encode_query(
    method: str,
    model: nn.Module,
    artifacts: dict[str, Any],
    query_pair: tuple,
    *,
    device: torch.device,
    noise_base: int = 0,
) -> dict[str, Any]:
    """Online query-side outputs stored for gallery scoring."""
    method = method.lower()
    cpu = torch.device("cpu")
    pre, post = pair_pre_post_lonlat12(
        query_pair,
        n_points=N_POINTS,
        noise_std_deg=NOISE_STD_DEG,
        noise_seed_pre=noise_base,
        noise_seed_post=noise_base + 1,
        device=cpu,
    )
    pre = pre.to(device)
    post = post.to(device)

    if method in ("traj2simvec", "t3s", "neutraj", "trajcl"):
        if method == "trajcl":
            _sync_trajcl_device(device)
        return {"z_q": model.encode_pre(pre.unsqueeze(0)).squeeze(0)}
    if method == "trajgat":
        helper = artifacts["trajgat_helper"]
        g_pre, _ = batch_pairs_to_graphs(
            helper,
            [query_pair],
            n_points=N_POINTS,
            noise_std_deg=NOISE_STD_DEG,
            noise_base=noise_base,
            device=device,
        )
        return {"z_q": model.encode_pre(g_pre).squeeze(0)}
    if method == "st2vec":
        mapper = artifacts["mapper"]
        pre_ids, pre_te, _, _ = batch_pairs_to_st2vec_inputs(
            mapper,
            [query_pair],
            n_points=N_POINTS,
            noise_std_deg=NOISE_STD_DEG,
            noise_base=noise_base,
            embed_dim=EMBED_DIM,
            device=device,
            max_node_idx=artifacts["max_node_idx"],
        )
        return {"z_q": model.encode_pre(pre_ids, pre_te).squeeze(0)}
    if method == "dan":
        return {"u_q": model.backbone(pre.unsqueeze(0)).squeeze(0)}
    if method in ("tat", "tat_graph"):
        return {"h_q": model.encoder_pre(pre.unsqueeze(0)).squeeze(0)}
    if method == "attnmove":
        return {"pred": model(pre.unsqueeze(0)).squeeze(0)}
    raise ValueError(method)


@torch.no_grad()
def score_gallery(
    method: str,
    model: nn.Module,
    query_state: dict[str, Any],
    gallery_data: dict[str, Any],
    *,
    device: torch.device,
    gallery_batch: int,
) -> torch.Tensor:
    method = method.lower()
    if method in TENSOR_REPR:
        return batched_cdist_scores(query_state["z_q"], gallery_data["embeddings"])
    if method == "dan":
        return _dan_score_gallery(model, query_state["u_q"], gallery_data["post_vecs"])
    if method in ("tat", "tat_graph"):
        return _tat_score_gallery(model, query_state["h_q"], gallery_data["post_vecs"])
    if method == "attnmove":
        return completion_gallery_scores(query_state["pred"], gallery_data["post_trajs"])
    raise ValueError(method)


@torch.no_grad()
def gallery_encode_and_score(
    method: str,
    model: nn.Module,
    artifacts: dict[str, Any],
    pairs: list[tuple],
    gallery_indices: list[int],
    query_state: dict[str, Any],
    *,
    device: torch.device,
    gallery_batch: int,
) -> torch.Tensor:
    """Online gallery: encode all N posts then score (batched)."""
    method = method.lower()
    if method == "attnmove":
        posts = _stack_lonlat_posts(pairs, gallery_indices, device)
        return completion_gallery_scores(query_state["pred"], posts)
    emb = encode_posts_batched(
        method, model, artifacts, pairs, gallery_indices, device=device
    )
    if method in TENSOR_REPR:
        return batched_cdist_scores(query_state["z_q"], emb)
    if method == "dan":
        return _dan_score_gallery(model, query_state["u_q"], emb)
    if method in ("tat", "tat_graph"):
        return _tat_score_gallery(model, query_state["h_q"], emb)
    raise ValueError(method)


def _dan_score_gallery(model: DANMotTrajSimplified, u_q: torch.Tensor, u_gallery: torch.Tensor) -> torch.Tensor:
    if getattr(model, "use_unmatched_dim", False) and hasattr(model, "score_query_gallery"):
        return model.score_query_gallery(u_q, u_gallery)
    n = u_gallery.shape[0]
    pair = torch.cat([u_q.unsqueeze(0).expand(n, -1), u_gallery], dim=-1)
    return model.pair_mlp(model.pair_ln(pair)).squeeze(-1)


def _tat_score_gallery(model: TATTrajTwoSegmentBaseline, h_q: torch.Tensor, h_gallery: torch.Tensor) -> torch.Tensor:
    n = h_gallery.shape[0]
    if not model.use_graph_update:
        pair = torch.cat([h_q.unsqueeze(0).expand(n, -1), h_gallery], dim=-1)
        return model.assoc_mlp(model.pair_ln(pair)).squeeze(-1)
    h_pre = h_q.unsqueeze(0).expand(n, -1)
    nodes = torch.stack([h_pre, h_gallery], dim=1)
    nodes = model.node_update(nodes)
    hp, hg = nodes[:, 0], nodes[:, 1]
    pair = torch.cat([hp, hg], dim=-1)
    return model.assoc_mlp(model.pair_ln(pair)).squeeze(-1)


def completion_gallery_scores(pred: torch.Tensor, gallery_post: torch.Tensor) -> torch.Tensor:
    """pred (T,2), gallery (N,T,2) -> (N,) higher=better."""
    diff = pred.unsqueeze(0) - gallery_post
    return -torch.linalg.vector_norm(diff, dim=-1).mean(dim=-1)


def count_aux_params(method: str, artifacts: dict[str, Any], model: nn.Module) -> int:
    method = method.lower()
    n = 0
    if method == "trajgat" and "pre_embedding" in artifacts:
        n += int(artifacts["pre_embedding"].numel())
    if method == "trajcl":
        n += int(model.cell_emb.weight.numel())  # type: ignore[attr-defined]
    if method == "st2vec":
        n += int(artifacts["road"].x.numel())
    if method == "attnmove":
        n += int(artifacts["region_vocab"].id2centroid.numel())
        n += int(artifacts["region_vocab"].dist_matrix.numel())
    return n
