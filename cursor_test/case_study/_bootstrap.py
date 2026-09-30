"""Self-contained import bootstrap for the case_study package.

All model code lives under ``cursor_test/case_study/vendor/`` (copies of
``cursor_baseline`` and ``cursor_mod_4``). Importing this module prepends the
vendor dir to ``sys.path`` so that ``import cursor_baseline`` / ``import
cursor_mod_4`` / ``import model_lib`` / ``import cfg`` resolve to the vendored
copies, NOT the original packages elsewhere in the repo.

Each entry-point script (workers, compute_selections) must ``import _bootstrap``
as its first project import.

NOTE on ``cfg``: the two stacks each register a *different* top-level ``cfg``
module (``cursor_baseline.model_lib.cfg`` vs ``cursor_mod_4.cfg``). They MUST
NOT be imported in the same process. Run the baseline worker and the mod4
worker as separate subprocesses.
"""

from __future__ import annotations

import pathlib
import sys

VENDOR = pathlib.Path(__file__).resolve().parent / "vendor"

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
