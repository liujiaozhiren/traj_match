"""
Load repo-root `model/ST2Vec` modules by file path so imports work without `model` being an
installable package (and avoid shadowing by any third-party `model` on sys.path).

`train_repr_baseline` must set ``sys.modules['cfg']`` before these loaders run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    if not path.is_file():
        raise FileNotFoundError(f"ST2Vec vendor file missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_st_encoder_class():
    """``model.ST2Vec.model.model_network.ST_Encoder``."""
    mod = _load_module(
        "st2vec_repo_model_network",
        _REPO_ROOT / "model" / "ST2Vec" / "model" / "model_network.py",
    )
    return mod.ST_Encoder


def get_build_graph_from_trajs() -> Callable[..., Any]:
    """``model.ST2Vec.model.traj2node.build_graph_from_trajs``."""
    mod = _load_module(
        "st2vec_repo_traj2node",
        _REPO_ROOT / "model" / "ST2Vec" / "model" / "traj2node.py",
    )
    return mod.build_graph_from_trajs
