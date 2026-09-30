#!/usr/bin/env python3
"""Run all tiers for one sweep (or all sweeps)."""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import subprocess
import sys
from pathlib import Path

from hyper_config import SWEEP_ORDER

_SCRIPTS = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=str, default=None, help="One sweep id; default all in SWEEP_ORDER")
    p.add_argument("--tiers", type=str, default="1,2,3,4,5")
    p.add_argument("--valid-only", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-root", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    sweeps = [args.sweep] if args.sweep else list(SWEEP_ORDER)
    tiers = [int(x) for x in args.tiers.split(",") if x.strip()]
    py = sys.executable
    for sid in sweeps:
        for tier in tiers:
            cmd = [py, str(_SCRIPTS / "run_tier.py"), "--sweep", sid, "--tier", str(tier)]
            if args.valid_only:
                cmd.append("--valid-only")
            if args.device:
                cmd += ["--device", args.device]
            if args.out_root:
                cmd += ["--out-root", args.out_root]
            print(f"[run_sweep] >>> {sid} tier {tier}", flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
