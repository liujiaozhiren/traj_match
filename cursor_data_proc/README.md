# cursor_data_proc

A self-contained copy of the `cursor_mod_4` pair-generation pipeline.

Copied from:
- `cursor_mod_4/proc.py` -> `cursor_data_proc/proc.py`
- `cursor_mod_4/cfg.py`  -> `cursor_data_proc/cfg.py` (only `con_file` default path adjusted)
- `cursor_mod_4/entrypoints/*` -> `cursor_data_proc/entrypoints/*`

## Default input

`cursor_data_proc/cfg.py` points `con_file` to `raw_data_proc/traj_cmb.pkl`.

## Generate v1 pairs

From repo root:

```bash
conda run --no-capture-output -n traj_match \
  python cursor_data_proc/entrypoints/dump_all_pairs.py \
  --output cursor_data_proc/out/all_pairs_v1.pkl
```

## Generate v2 maxdtw (from v1)

```bash
conda run --no-capture-output -n traj_match \
  python cursor_data_proc/entrypoints/gen_noisy_pairs_v2.py \
  --input cursor_data_proc/out/all_pairs_v1.pkl \
  --output-dir cursor_data_proc/out \
  --schemes maxdtw
```
