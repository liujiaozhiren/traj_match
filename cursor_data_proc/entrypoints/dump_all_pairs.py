"""
Dump `all_pairs = clean_pairs(mul_p + sing_p)` to a pickle file.

This is intended to snapshot the "old version" of the dataset produced from
`pipeline/traj_cmb.pkl` using the current cursor_mod_4 preprocessing code path.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_rp = str(_ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

sys.modules["cfg"] = importlib.import_module("cursor_data_proc.cfg")
import cfg  # noqa: E402

from cursor_data_proc.proc import clean_pairs, proc_multi_single_traj  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pkl path. Default: cursor_mod_4/pipeline/all_pairs_v1.pkl",
    )
    p.add_argument(
        "--max-raw-trajs",
        type=int,
        default=None,
        help="Optional: only use first N raw single trajectories (debug).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.output or os.path.join(_ROOT, "cursor_data_proc", "out", "all_pairs_v1.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    if args.max_raw_trajs is not None:
        single_traj_cmb = single_traj_cmb[: int(args.max_raw_trajs)]

    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    all_pairs = clean_pairs(mul_p + sing_p)

    pickle.dump(all_pairs, open(out_path, "wb"))
    print(f"[dump_all_pairs] cfg.con_file={cfg.con_file}", flush=True)
    print(f"[dump_all_pairs] saved -> {out_path}", flush=True)
    print(f"[dump_all_pairs] num_pairs={len(all_pairs)}", flush=True)
    if all_pairs:
        item = all_pairs[0]
        mid, A, B, L_A, L_B, F_A, F_B = item
        print(
            f"[dump_all_pairs] sample0 lens: A={len(A)} B={len(B)} L_A={len(L_A)} L_B={len(L_B)} F_A={len(F_A)} F_B={len(F_B)}",
            flush=True,
        )
        print(f"[dump_all_pairs] sample0 mid={mid}", flush=True)


if __name__ == "__main__":
    main()

