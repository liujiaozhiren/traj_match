"""
NeuTraj-style twin encoders (pre/post) using ``geo_rnns/neutraj_model.RNNEncoder`` or stacked SAM cells.

- ``stard_lstm=True``: standard LSTM/GRU on normalized lon/lat (channels 3–4 padded).
- ``stard_lstm=False``: ``SAM_LSTMCell`` with discrete grid indices in the last two channels.
"""

from __future__ import annotations

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
    """Patch ``tools/config.py`` before constructing NeuTraj cells."""
    _ensure_neutraj_on_path()
    import tools.config as nc  # noqa: E402

    nc.device = "cuda" if device_torch.type == "cuda" else "cpu"
    nc.gird_size = [int(grid_size[0]), int(grid_size[1])]
    nc.recurrent_unit = recurrent_unit
    nc.stard_unit = stard_unit
    nc.incell = incell
    nc.spatial_width = int(spatial_width)


def lonlat_mean_std_from_pairs(pairs: list, max_pairs: int = 4000) -> tuple[torch.Tensor, torch.Tensor]:
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


def lonlat_bounds_from_pairs(pairs: list, max_pairs: int = 4000) -> tuple[float, float, float, float]:
    lons: list[float] = []
    lats: list[float] = []
    for item in pairs[:max_pairs]:
        _, A, B, *_rest = item
        for traj in (A, B):
            for p in traj:
                lons.append(float(p[1]))
                lats.append(float(p[2]))
    if not lons:
        return 0.0, 1.0, 0.0, 1.0
    return min(lons), max(lons), min(lats), max(lats)


def _lonlat_to_neutraj_inputs(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    bounds: tuple[float, float, float, float] | None,
    grid_size: tuple[int, int],
    use_grid: bool,
) -> torch.Tensor:
    """(B, L, 2) lon/lat -> (B, L, 4)."""
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    xn = (x - mean.view(1, 1, 2)) / std.view(1, 1, 2)
    if not use_grid:
        z = torch.zeros(x.shape[0], x.shape[1], 2, device=x.device, dtype=x.dtype)
        return torch.cat([xn, z], dim=-1)

    min_lon, max_lon, min_lat, max_lat = bounds or (0.0, 1.0, 0.0, 1.0)
    gw, gh = int(grid_size[0]), int(grid_size[1])
    lon = x[..., 0]
    lat = x[..., 1]
    gx = ((lon - min_lon) / (max_lon - min_lon + 1e-8) * (gw - 1)).round().clamp(0, gw - 1)
    gy = ((lat - min_lat) / (max_lat - min_lat + 1e-8) * (gh - 1)).round().clamp(0, gh - 1)
    grid = torch.stack([gx, gy], dim=-1)
    return torch.cat([xn, grid], dim=-1)


class StackedSAMRNNEncoder(nn.Module):
    """``num_layers`` SAM LSTM cells sharing one spatial memory."""

    def __init__(
        self,
        hidden_size: int,
        grid_size: list[int],
        *,
        num_layers: int = 6,
        incell: bool = True,
        spatial_embedding=None,
        atten=None,
    ):
        super().__init__()
        _ensure_neutraj_on_path()
        from geo_rnns.sam_cells import SAM_LSTMCell  # noqa: E402

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        cells = []
        for _ in range(self.num_layers):
            cells.append(
                SAM_LSTMCell(
                    4,
                    hidden_size,
                    grid_size,
                    incell=incell,
                    spatial_embedding=spatial_embedding,
                    atten=atten,
                )
            )
        self.cells = nn.ModuleList(cells)
        self.layer_in = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(hidden_size, 2) for _ in range(self.num_layers - 1)]
        )

    def forward(self, x4: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        b, L, _ = x4.shape
        grid = x4[..., 2:4]
        layer_h = [torch.zeros(b, self.hidden_size, device=x4.device, dtype=x4.dtype) for _ in range(self.num_layers)]
        layer_c = [torch.zeros(b, self.hidden_size, device=x4.device, dtype=x4.dtype) for _ in range(self.num_layers)]
        outputs: list[torch.Tensor] = []
        for t in range(L):
            g_t = grid[:, t, :]
            for li, cell in enumerate(self.cells):
                feat = x4[:, t, :2] if li == 0 else self.layer_in[li](layer_h[li])
                inp = torch.cat([feat, g_t], dim=-1)
                layer_h[li], layer_c[li] = cell(inp, (layer_h[li], layer_c[li]))
            outputs.append(layer_h[-1])
        out = torch.stack(outputs, dim=1)
        mask_out = []
        for bi, v in enumerate(lengths):
            mask_out.append(out[bi, v - 1].view(1, -1))
        return torch.cat(mask_out, dim=0)


class NeuTrajTwin(nn.Module):
    """Separate pre/post encoders; embeddings (B, target_size)."""

    def __init__(
        self,
        target_size: int,
        lonlat_mean: torch.Tensor,
        lonlat_std: torch.Tensor,
        *,
        grid_size: tuple[int, int] = (500, 500),
        lonlat_bounds: tuple[float, float, float, float] | None = None,
        recurrent_unit: str = "LSTM",
        stard_lstm: bool = True,
        num_layers: int = 1,
        incell: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.target_size = target_size
        self.stard_lstm = bool(stard_lstm)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self._recurrent_unit = recurrent_unit
        dev = device or torch.device("cpu")
        sync_neutraj_tools_config(
            device_torch=dev,
            grid_size=self.grid_size,
            recurrent_unit=recurrent_unit,
            stard_unit=stard_lstm,
            incell=incell,
        )
        _ensure_neutraj_on_path()

        if self.stard_lstm:
            from geo_rnns.neutraj_model import RNNEncoder  # noqa: E402

            self.enc_pre = RNNEncoder(4, target_size, list(grid_size), stard_LSTM=True, incell=incell)
            self.enc_post = RNNEncoder(4, target_size, list(grid_size), stard_LSTM=True, incell=incell)
        else:
            if recurrent_unit != "LSTM":
                raise ValueError("SAM stacked encoder currently supports recurrent_unit=LSTM only")
            _ensure_neutraj_on_path()
            import tools.config as nc  # noqa: E402
            from geo_rnns.memory import Attention, SpatialExternalMemory  # noqa: E402

            sw = int(nc.spatial_width)
            mem_n = self.grid_size[0] + 3 * sw
            mem_m = self.grid_size[1] + 3 * sw
            if dev.type == "cuda":
                shared_mem = SpatialExternalMemory(mem_n, mem_m, target_size).cuda()
                shared_atten = Attention(target_size).cuda()
            else:
                shared_mem = SpatialExternalMemory(mem_n, mem_m, target_size).cpu()
                shared_atten = Attention(target_size).cpu()
            sam_kw = dict(
                num_layers=int(num_layers),
                incell=incell,
                spatial_embedding=shared_mem,
                atten=shared_atten,
            )
            self.enc_pre = StackedSAMRNNEncoder(target_size, list(grid_size), **sam_kw)
            self.enc_post = StackedSAMRNNEncoder(target_size, list(grid_size), **sam_kw)

        self.register_buffer("lonlat_mean", lonlat_mean.clone())
        self.register_buffer("lonlat_std", lonlat_std.clone())
        if lonlat_bounds is not None:
            min_lon, max_lon, min_lat, max_lat = lonlat_bounds
            self.register_buffer("lonlat_bounds", torch.tensor([min_lon, max_lon, min_lat, max_lat]))
        else:
            self.register_buffer("lonlat_bounds", torch.tensor([0.0, 1.0, 0.0, 1.0]))

    def _to_inputs(self, x: torch.Tensor) -> torch.Tensor:
        bounds = tuple(float(v) for v in self.lonlat_bounds.tolist())
        return _lonlat_to_neutraj_inputs(
            x,
            self.lonlat_mean,
            self.lonlat_std,
            bounds=bounds,
            grid_size=self.grid_size,
            use_grid=not self.stard_lstm,
        )

    def _encode_rnn(self, enc: nn.Module, x4: torch.Tensor) -> torch.Tensor:
        B, L, _ = x4.shape
        lengths = [int(L)] * B
        if self.stard_lstm:
            if self._recurrent_unit in ("GRU", "SimpleRNN"):
                h0 = torch.zeros(B, self.target_size, device=x4.device, dtype=x4.dtype)
                return enc([x4, lengths], h0)
            h0 = torch.zeros(B, self.target_size, device=x4.device, dtype=x4.dtype)
            c0 = torch.zeros(B, self.target_size, device=x4.device, dtype=x4.dtype)
            return enc([x4, lengths], (h0, c0))
        return enc(x4, lengths)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_rnn(self.enc_pre, self._to_inputs(x))

    def encode_post(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_rnn(self.enc_post, self._to_inputs(x))

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_pre(pre), self.encode_post(post)
