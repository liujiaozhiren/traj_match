"""
NeuTraj-style twin encoders (pre/post) using `model/neutraj/geo_rnns/neutraj_model.RNNEncoder`.

We default to **stard_LSTM=True** (standard LSTM/GRU on the first two input channels), matching the
original NeuTraj path where each step uses normalized lon/lat only; the last two channels are
padding. This avoids wiring the full spatial grid + SAM memory for the baseline while still using
the same RNN stack as the copied NeuTraj code.

Set ``stard_lstm=False`` to use SAM-augmented cells; then the last two channels must be valid
discrete grid indices (see NeuTraj data pipeline).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_NEUTRAJ_ROOT = Path(__file__).resolve().parent / "model_lib" / "neutraj"


def _ensure_neutraj_on_path() -> None:
    r = str(_NEUTRAJ_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def sync_neutraj_tools_config(
    *,
    device_torch: torch.device,
    grid_size: tuple[int, int],
    recurrent_unit: str,
    stard_unit: bool,
    incell: bool,
    spatial_width: int = 2,
) -> None:
    """Patch `model/neutraj/tools/config.py` before constructing RNNEncoder (matches NeuTraj expectations)."""
    _ensure_neutraj_on_path()
    import tools.config as nc  # noqa: E402

    nc.device = "cuda" if device_torch.type == "cuda" else "cpu"
    nc.gird_size = [int(grid_size[0]), int(grid_size[1])]
    nc.recurrent_unit = recurrent_unit
    nc.stard_unit = stard_unit
    nc.incell = incell
    nc.spatial_width = int(spatial_width)


def lonlat_mean_std_from_pairs(pairs: list, max_pairs: int = 4000) -> tuple[torch.Tensor, torch.Tensor]:
    """Rough global lon/lat mean/std over a prefix of train pairs (same convention as lonlat_traj)."""
    xs: list[float] = []
    ys: list[float] = []
    for item in pairs[:max_pairs]:
        _, A, B, *_rest = item
        for traj in (A, B):
            for p in traj:
                xs.append(float(p[1]))
                ys.append(float(p[2]))
    if not xs:
        return torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs) / len(xs)
    vy = sum((y - my) ** 2 for y in ys) / len(ys)
    sx = vx**0.5 + 1e-6
    sy = vy**0.5 + 1e-6
    return torch.tensor([mx, my], dtype=torch.float32), torch.tensor([sx, sy], dtype=torch.float32)


def _lonlat_to_neutraj_inputs(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """(B, L, 2) lon/lat degrees -> (B, L, 4) NeuTraj tensor (stard path uses first two dims)."""
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    xn = (x - mean.view(1, 1, 2)) / std.view(1, 1, 2)
    z = torch.zeros(x.shape[0], x.shape[1], 2, device=x.device, dtype=x.dtype)
    return torch.cat([xn, z], dim=-1)


class NeuTrajTwin(nn.Module):
    """Separate pre/post RNNEncoder instances; embeddings (B, target_size)."""

    def __init__(
        self,
        target_size: int,
        lonlat_mean: torch.Tensor,
        lonlat_std: torch.Tensor,
        *,
        grid_size: tuple[int, int] = (500, 500),
        recurrent_unit: str = "LSTM",
        stard_lstm: bool = True,
        incell: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        if not stard_lstm:
            raise ValueError(
                "This baseline only supports stard_lstm=True (standard LSTM/GRU on normalized lon/lat). "
                "SAM cells need NeuTraj grid indices in the last two channels; use the full neutraj train.py pipeline."
            )
        self.target_size = target_size
        dev = device or torch.device("cpu")
        sync_neutraj_tools_config(
            device_torch=dev,
            grid_size=grid_size,
            recurrent_unit=recurrent_unit,
            stard_unit=stard_lstm,
            incell=incell,
        )
        _ensure_neutraj_on_path()
        from geo_rnns.neutraj_model import RNNEncoder  # noqa: E402

        self.enc_pre = RNNEncoder(4, target_size, list(grid_size), stard_LSTM=stard_lstm, incell=incell)
        self.enc_post = RNNEncoder(4, target_size, list(grid_size), stard_LSTM=stard_lstm, incell=incell)
        self.register_buffer("lonlat_mean", lonlat_mean.clone())
        self.register_buffer("lonlat_std", lonlat_std.clone())
        self._recurrent_unit = recurrent_unit

    def _encode_rnn(self, enc: nn.Module, x4: torch.Tensor) -> torch.Tensor:
        B, L, _ = x4.shape
        lengths = [int(L)] * B
        if self._recurrent_unit in ("GRU", "SimpleRNN"):
            h0 = torch.zeros(B, self.target_size, device=x4.device, dtype=x4.dtype)
            return enc([x4, lengths], h0)
        h0 = torch.zeros(B, self.target_size, device=x4.device, dtype=x4.dtype)
        c0 = torch.zeros(B, self.target_size, device=x4.device, dtype=x4.dtype)
        return enc([x4, lengths], (h0, c0))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        x4 = _lonlat_to_neutraj_inputs(x, self.lonlat_mean, self.lonlat_std)
        return self._encode_rnn(self.enc_pre, x4)

    def encode_post(self, x: torch.Tensor) -> torch.Tensor:
        x4 = _lonlat_to_neutraj_inputs(x, self.lonlat_mean, self.lonlat_std)
        return self._encode_rnn(self.enc_post, x4)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_pre(pre), self.encode_post(post)
