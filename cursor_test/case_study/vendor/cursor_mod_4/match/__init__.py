"""
Self-contained match package for cursor_mod_4.

Schemes:
  - 0: original GMN + Hungarian (migrated from match_model/new_test/pipeline)
  - 1: Sinkhorn assignment (soft -> hard) + GMN
  - 2: Cross-attn topk (soft -> hard) + GMN (C1)
  - 3: SetTransformer-style set-to-set matcher (E2)
  - 4: Pair-Transformer end-to-end matcher (E3)
"""

from .matcher import MatchModel, MatchInputs, build_match_inputs

__all__ = ["MatchModel", "MatchInputs", "build_match_inputs"]

