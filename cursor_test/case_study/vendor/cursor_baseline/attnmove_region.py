"""Discrete region vocabulary for AttnMove-style completion (fit on train pairs)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cursor_baseline.completion_traj import pair_to_ab_lonlat

PAD_ID = 0
UNK_ID = 1
_REGION0 = 2


@dataclass
class RegionVocab:
    """Grid-quantized regions with centroids and pairwise distances (degrees)."""

    lng_ld: float
    lat_ld: float
    cell_deg: float
    grid2id: dict[str, int]
    id2centroid: torch.Tensor  # (V, 2) lon/lat on CPU
    dist_matrix: torch.Tensor  # (V, V) float32 on CPU

    @property
    def vocab_size(self) -> int:
        return int(self.id2centroid.shape[0])

    @classmethod
    def fit_from_pairs(
        cls,
        pairs: list,
        *,
        cpu: torch.device,
        cell_deg: float = 0.0002,
    ) -> RegionVocab:
        coords: list[tuple[float, float]] = []
        for p in pairs:
            pr = pair_to_ab_lonlat(p, cpu)
            if pr is None:
                continue
            a, b = pr
            for t in range(a.shape[0]):
                coords.append((float(a[t, 0].item()), float(a[t, 1].item())))
                coords.append((float(b[t, 0].item()), float(b[t, 1].item())))
        if not coords:
            raise RuntimeError("RegionVocab.fit_from_pairs: no coordinates")
        lngs = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        lng_ld = min(lngs)
        lat_ld = min(lats)
        grid2id: dict[str, int] = {}
        grid_sums: dict[str, list[float]] = {}
        grid_cnt: dict[str, int] = {}
        next_id = _REGION0

        def _accum(lng: float, lat: float) -> None:
            nonlocal next_id
            g = _grid_key(lng, lat, lng_ld, lat_ld, cell_deg)
            if g not in grid2id:
                grid2id[g] = next_id
                next_id += 1
                grid_sums[g] = [0.0, 0.0]
                grid_cnt[g] = 0
            grid_sums[g][0] += lng
            grid_sums[g][1] += lat
            grid_cnt[g] += 1

        for lng, lat in coords:
            _accum(lng, lat)

        n_reg = next_id
        id2c = torch.zeros(n_reg, 2, dtype=torch.float32)
        id2c[PAD_ID] = 0.0
        id2c[UNK_ID] = 0.0
        for g, rid in grid2id.items():
            c = grid_cnt[g]
            id2c[rid, 0] = grid_sums[g][0] / c
            id2c[rid, 1] = grid_sums[g][1] / c

        dist = torch.cdist(id2c, id2c, p=2)

        return cls(
            lng_ld=float(lng_ld),
            lat_ld=float(lat_ld),
            cell_deg=float(cell_deg),
            grid2id=grid2id,
            id2centroid=id2c,
            dist_matrix=dist,
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        payload = {
            "lng_ld": self.lng_ld,
            "lat_ld": self.lat_ld,
            "cell_deg": self.cell_deg,
            "grid2id": self.grid2id,
            "id2centroid": self.id2centroid.tolist(),
            "dist_matrix": self.dist_matrix.tolist(),
        }
        p.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RegionVocab:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        id2c = torch.tensor(payload["id2centroid"], dtype=torch.float32)
        dist = torch.tensor(payload["dist_matrix"], dtype=torch.float32)
        return cls(
            lng_ld=float(payload["lng_ld"]),
            lat_ld=float(payload["lat_ld"]),
            cell_deg=float(payload["cell_deg"]),
            grid2id={str(k): int(v) for k, v in payload["grid2id"].items()},
            id2centroid=id2c,
            dist_matrix=dist,
        )

    def _lonlat_to_grid(self, lng: float, lat: float) -> str:
        return _grid_key(lng, lat, self.lng_ld, self.lat_ld, self.cell_deg)

    def _resolve_grid(self, grid: str) -> int:
        if grid in self.grid2id:
            return self.grid2id[grid]
        parts = grid.split("-")
        if len(parts) != 2:
            return UNK_ID
        gx, gy = int(parts[0]), int(parts[1])
        for step in range(1, 64):
            for dx, dy in _neighbor_ring(gx, gy, step):
                g2 = f"{dx}-{dy}"
                if g2 in self.grid2id:
                    return self.grid2id[g2]
        return UNK_ID

    def lonlat_to_ids(self, xy: torch.Tensor) -> torch.Tensor:
        """xy: (..., 2) -> (...) long on same device."""
        flat = xy.reshape(-1, 2).detach().cpu()
        out: list[int] = []
        for i in range(flat.shape[0]):
            lng, lat = float(flat[i, 0].item()), float(flat[i, 1].item())
            g = self._lonlat_to_grid(lng, lat)
            out.append(self._resolve_grid(g))
        ids = torch.tensor(out, dtype=torch.long, device=xy.device).reshape(xy.shape[:-1])
        return ids

    def ids_to_lonlat(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (...) -> (..., 2) on ids.device."""
        flat = ids.reshape(-1).detach().cpu().tolist()
        cents = self.id2centroid
        rows = [cents[int(min(max(i, 0), cents.shape[0] - 1))] for i in flat]
        t = torch.stack(rows, dim=0).to(device=ids.device, dtype=torch.float32)
        return t.reshape(*ids.shape, 2)


def _grid_key(lng: float, lat: float, lng_ld: float, lat_ld: float, cell_deg: float) -> str:
    gx = int((lng - lng_ld) / cell_deg)
    gy = int((lat - lat_ld) / cell_deg)
    return f"{gx}-{gy}"


def _neighbor_ring(gx: int, gy: int, r: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dx in range(gx - r, gx + r + 1):
        for dy in range(gy - r, gy + r + 1):
            if max(abs(dx - gx), abs(dy - gy)) == r:
                out.append((dx, dy))
    return out
