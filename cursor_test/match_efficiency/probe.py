"""Timing and peak memory probes for one E2E retrieval request."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import psutil
import torch


@dataclass
class ProbeResult:
    wall_sec: float = 0.0
    peak_rss_mib: float = 0.0
    peak_vram_mib: float = 0.0


@dataclass
class E2ETiming:
    t_query: float = 0.0
    t_gallery: float = 0.0
    t_argmax: float = 0.0
    t_e2e: float = 0.0
    peak_rss_mib: float = 0.0
    peak_vram_mib: float = 0.0

    def as_dict(self) -> dict:
        return {
            "t_query": self.t_query,
            "t_gallery": self.t_gallery,
            "t_argmax": self.t_argmax,
            "t_e2e": self.t_e2e,
            "peak_rss_mib": self.peak_rss_mib,
            "peak_vram_mib": self.peak_vram_mib,
        }


class MemoryProbe:
    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self._rss0 = self._rss_mib()
        self._peak_rss = self._rss0
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def _rss_mib(self) -> float:
        return self._proc.memory_info().rss / (1024**2)

    def sample(self) -> None:
        self._peak_rss = max(self._peak_rss, self._rss_mib())

    def finish(self) -> tuple[float, float]:
        self.sample()
        vram = 0.0
        if self._use_cuda:
            torch.cuda.synchronize()
            vram = torch.cuda.max_memory_allocated() / (1024**2)
        return max(0.0, self._peak_rss - self._rss0), vram


@contextmanager
def timed_section() -> Iterator[list[float]]:
    buf: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield buf
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    buf.append(time.perf_counter() - t0)


def count_parameters(*modules, aux_tensors: list[torch.Tensor] | None = None) -> int:
    n = 0
    for m in modules:
        if m is None:
            continue
        if isinstance(m, torch.Tensor):
            if m.requires_grad:
                n += m.numel()
            continue
        if isinstance(m, torch.nn.Parameter):
            n += m.numel()
            continue
        n += sum(p.numel() for p in m.parameters())
    if aux_tensors:
        for t in aux_tensors:
            if t is not None and getattr(t, "requires_grad", False):
                n += t.numel()
    return n


def count_storage_elements(*tensors) -> int:
    return sum(int(t.numel()) for t in tensors if t is not None)
