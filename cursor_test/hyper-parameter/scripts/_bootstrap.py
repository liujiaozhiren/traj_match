"""Ensure `scripts/` is on sys.path so local modules resolve without importing mod4."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_p = str(_SCRIPTS)
if _p not in sys.path:
    sys.path.insert(0, _p)
